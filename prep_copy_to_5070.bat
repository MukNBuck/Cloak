@echo off
REM Windows helper if you stage files here before copying to the 5070 Linux box.
REM Actual CRAG runs are meant for the 5070 machine via setup_crag_5070.sh / run_crag_5070.sh.

echo Copy the Cloak folder + CRAG jsonl to the 5070 machine, then:
echo   cd Cloak
echo   bash setup_crag_5070.sh
echo   bash run_crag_5070.sh smoke
echo See CRAG_5070.md
pause
