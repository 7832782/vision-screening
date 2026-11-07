$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

$dir = 'F:\DeepSeek Harness\project-test\vision-screening\_test'
$dir = 'F:\DeepSeek Harness\' + [char]0x9879 + [char]0x76EE + '\vision-screening\_test'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Get-ChildItem $dir -Filter *.wav -ErrorAction SilentlyContinue | Remove-Item -Force

$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

# shang / xia / zuo / you
$words = [ordered]@{
    up    = [string][char]0x4E0A
    down  = [string][char]0x4E0B
    left  = [string][char]0x5DE6
    right = [string][char]0x53F3
}
$voices = @('Microsoft Huihui Desktop', 'Microsoft Kangkang', 'Microsoft Yaoyao')

foreach ($v in $voices) {
    $short = ($v -split ' ')[1]
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice($v)
    foreach ($k in $words.Keys) {
        foreach ($rate in -3, 0, 2, 5) {
            $s.Rate = $rate
            $path = Join-Path $dir ("{0}_{1}_{2}.wav" -f $k, $short, $rate)
            $s.SetOutputToWaveFile($path, $fmt)
            $s.Speak($words[$k])
            $s.SetOutputToNull()
        }
    }
    $s.Dispose()
}

Get-ChildItem $dir -Filter *.wav | Measure-Object -Property Length -Sum |
    ForEach-Object { "generated {0} wav, {1:N0} KB" -f $_.Count, ($_.Sum / 1KB) }
Get-ChildItem $dir -Filter *.wav | Select-Object -First 4 Name, Length | Format-Table -AutoSize
