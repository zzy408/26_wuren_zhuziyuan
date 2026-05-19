#!/bin/bash
mkdir -p linux_practice/{docs,backup}
cd linux_practice/docs
touch readme.txt notes.log temp.tmp
rm temp.tmp
mv notes.log daily_report.txt
echo "Project Status:Active" > daily_report.txt
echo $(date) >> daily_report.txt
cp *.txt /home/zhuziyuan/practice/linux_practice/backup
cd ..
cd backup
chmod  444 *
echo "Active Complete.File backup is read-only."
