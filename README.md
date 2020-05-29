# Test_Summarization-Sentiment_Analyze



### How to make a new branch
```
git branch your-name
```
### When you push, 
```
 git branch //This confirms that you are now working on your new branch
 git status
 git add -A
 git commit -m "Your commit message"
 git push origin urName-dev
```
### When you pull,
```
 git branch //it should still be your development branch, not master
 git checkout master
 git pull origin master
 git checkout urName-dev
 git merge master
```
