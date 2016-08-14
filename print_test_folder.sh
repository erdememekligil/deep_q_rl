$targets = @()
$folders = dir "D:\dev\projects\deep_q_rl\deep_q_rl\results" | Where {$_.mode -match "d"}
foreach ($folder in $folders) {
     if(
        ($folder.GetFiles() |
         Measure-Object |
         Select -ExpandProperty Count) -eq 4) 
        {$targets += $folder}
}
$targets | Format-Table -Property Name