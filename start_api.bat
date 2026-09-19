@echo off
chcp 65001
::set PYTHONUSERBASE=.\python\Lib\site-packages
::set PYTHONPATH=.\python\Lib\site-packages
::set PATH=%PATH%;.\python\Scripts
uv run python ServerApi.py --model-dir "D:\qwen3-tts-gguf\model-base" --port 9880

pause