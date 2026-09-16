# deploy.ps1 — hermesnote-v2 一鍵部署
# 每次都整組重新部署（frontend + backend + training 三個容器一起），不做局部判斷
# 目標路徑：NAS /mnt/Hermesnote/web/hermes（跟本機這個 repo 根目錄對稱）
#
# 設計原則（重要，不要改掉）：這支腳本只做「部署程式碼」，不負責容器的生死。
# 部署前會先確認三個容器本來就存在且在跑；只要有一個對不上，整支直接中止、
# 不會往下做任何事，更不會嘗試建立容器頂上去。容器層出問題是另一個要單獨排查
# 的問題，不該被這支腳本悄悄用「順手建一個新的」蓋過去。

$NAS_HOST = "192.168.0.44"
$NAS_HOSTNAME = "TRUENAS"
$NAS_SSH_USER = "truenas_admin"
$NAS_SMB_USER = "hermes"
$NAS_HERMES_PATH = "/mnt/Hermesnote/web/hermes"
$SMB_SHARE = "\\$NAS_HOSTNAME\web"
$SMB_HERMES = "$SMB_SHARE\hermes"
$LOCAL_ROOT = "$PSScriptRoot"

# 確保 SMB 已掛載（已連線就跳過）
$existing = net use 2>$null | Select-String "TRUENAS"
if (-not $existing) {
    Write-Host ">>> Mounting SMB share..." -ForegroundColor Cyan
    $smbPwd = Read-Host "Enter SMB password for $NAS_SMB_USER" -AsSecureString
    $smbPwdPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($smbPwd)
    )
    net use $SMB_SHARE /user:$NAS_SMB_USER $smbPwdPlain | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "SMB mount failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ">>> SMB already connected, skipping mount" -ForegroundColor DarkGray
}

Write-Host ">>> Checking existing containers are up before touching anything..." -ForegroundColor Cyan
$expected = @("hermesnote-frontend", "hermesnote-backend", "hermesnote-training")
$checkCmd = "sudo docker inspect --format '{{.Name}}:{{.State.Running}}' " + ($expected -join " ") + " 2>&1"
$checkResult = ssh "$NAS_SSH_USER@$NAS_HOST" $checkCmd
Write-Host $checkResult
$checkText = $checkResult -join "`n"
$allRunning = $true
foreach ($name in $expected) {
    if ($checkText -notmatch "/${name}:true") {
        $allRunning = $false
    }
}
if (-not $allRunning) {
    Write-Host "容器狀態不對（缺少或沒在跑），停止部署。這是容器層的問題，先排查，不會建立新容器。" -ForegroundColor Red
    exit 1
}
Write-Host ">>> Containers OK, proceeding" -ForegroundColor Green

Write-Host ">>> [Frontend] Building..." -ForegroundColor Cyan
Push-Location "$LOCAL_ROOT\frontend"
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Frontend build failed" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location

Write-Host ">>> Copying repo to NAS..." -ForegroundColor Cyan
robocopy "$LOCAL_ROOT\frontend\dist" "$SMB_HERMES\frontend\dist" /MIR /NFL /NDL /NJH /NJS
robocopy "$LOCAL_ROOT\backend" "$SMB_HERMES\backend" /MIR /XD __pycache__ .git .venv /XF "*.pyc" "*.pyo" /NFL /NDL /NJH /NJS
Copy-Item "$LOCAL_ROOT\docker-compose.yml" "$SMB_HERMES\docker-compose.yml" -Force
Copy-Item "$LOCAL_ROOT\nginx.conf" "$SMB_HERMES\nginx.conf" -Force
if ($LASTEXITCODE -ge 8) {
    Write-Host "Copy to NAS failed" -ForegroundColor Red
    exit 1
}
Write-Host ">>> Copy done" -ForegroundColor Green

Write-Host ">>> Updating containers with new code via SSH (docker compose)..." -ForegroundColor Cyan
# 前面已經確認過三個容器都存在且在跑，這裡的 up --build 保證只是「換新 image 更新」，
# 不會是「從無到有建立」——那個情境在上面的檢查就會先擋下來、整支腳本已經 exit 1 了
#
# frontend 服務沒有 build 步驟，內容全靠 bind mount（dist/、nginx.conf），
# 光是 host 上的檔案變了，docker compose 不會因此重建/重啟這個容器——
# nginx process 會繼續用它啟動當下讀進去的舊設定，完全不知道檔案已經換了。
# 所以每次都額外明講 restart frontend，強制它重新讀取新的 nginx.conf/靜態檔案。
# 分開 build / up 兩步（而不是 up --build 一行），這樣 build 失敗會在這裡直接被 SSH 的
# exit code 抓到、腳本就會停下來報錯——之前 up --build 合成一行，build 卡住也不會讓
# 整條 ssh 指令回傳非 0，腳本會誤判成功，這是 2026-09-08 深夜實際發生過的問題
$sshCmd = "cd $NAS_HERMES_PATH && sudo docker compose build backend training && sudo docker compose up -d && sudo docker compose restart frontend && sudo docker compose ps"
ssh "$NAS_SSH_USER@$NAS_HOST" $sshCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host ">>> Deploy 失敗！SSH 那段指令回傳非 0，往上找是 build 還是 up 出錯" -ForegroundColor Red
    exit 1
}
Write-Host ">>> Deploy complete! 請對照上面 docker compose ps 印出的 CREATED 欄位，確認 backend/training 是不是剛剛才重建的，不是舊的" -ForegroundColor Green

