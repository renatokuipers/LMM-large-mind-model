@echo off
echo Installing LMM Project in development mode...
pip install -e .

echo Running LMM Project...
cd lmm_project
python main.py %* 