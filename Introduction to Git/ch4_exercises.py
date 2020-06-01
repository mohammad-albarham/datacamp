# Exercise_1 
#1
git checkout summary-statistics
#2
git rm report.txt
#3
git commit -m  "Removing report"
#4
ls
#5
git checkout master
#6
ls


--------------------------------------------------
# Exercise_2 
#1
git checkout -b deleting-report
#2
git rm report.txt
#3
git commit -m "New chenges"
#4
git diff master..deleting-report


--------------------------------------------------
# Exercise_3 
#1
git merge summary-statistics master


--------------------------------------------------
# Exercise_4 
#1
git merge alter-report-title  master
#2
git status
#3
nano report.txt
#4
git add report.txt
#5
git commit


--------------------------------------------------
