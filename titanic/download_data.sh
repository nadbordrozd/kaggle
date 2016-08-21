%%bash
curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/5261/people.csv.zip?sv=2012-02-12&se=2016-08-24T19%3A47%3A51Z&sr=b&sp=r&sig=KcGIYQ%2BX0r7XtGa%2FiZQGfG4oXzHyvFJnPK91Se0UvKw%3D' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/predicting-red-hat-business-value/data' -H 'Connection: keep-alive' --compressed > data/people.csv.gz

%%bash
curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/5261/sample_submission.csv.zip?sv=2012-02-12&se=2016-08-24T19%3A52%3A04Z&sr=b&sp=r&sig=jRhAdux%2FZcY5e13ST%2Fmm9BdTuhAjvxuyIU%2BCX2HymgU%3D' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/predicting-red-hat-business-value/data' -H 'Connection: keep-alive' --compressed > data/sample_submission.csv.gz

curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/5261/act_test.csv.zip?sv=2012-02-12&se=2016-08-24T19%3A57%3A03Z&sr=b&sp=r&sig=M75uNAzlPGC5IVSEAiarEj4JsfuZFJP8n7Tys6Kb%2F6w%3D' -H 'Referer: https://www.kaggle.com/c/predicting-red-hat-business-value/data' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36' --compressed > data/act_test.csv.gz

curl 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/5261/act_train.csv.zip?sv=2012-02-12&se=2016-08-24T19%3A57%3A35Z&sr=b&sp=r&sig=uN3%2B0q8J0qg7stz5%2FhFqywbY%2BNwR1URoDFxbl2aibYI%3D' -H 'Accept-Encoding: gzip, deflate, sdch, br' -H 'Accept-Language: en-US,en;q=0.8' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Referer: https://www.kaggle.com/c/predicting-red-hat-business-value/data' -H 'Connection: keep-alive' --compressed > data/act_train.csv.gz


gunzip data/people.csv.gz
gunzip data/sample_submission.csv.gz
gunzip data/act_test.csv.gz
gunzip data/act_train.csv.gz