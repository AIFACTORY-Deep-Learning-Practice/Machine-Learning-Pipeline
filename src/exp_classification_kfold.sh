echo "Logistic Regression"
python main.py --task classification --model Logistic --kfold 10 && \
echo "Ridge Regression"
python main.py --task classification --model Ridge --kfold 10 && \
echo "Decision Tree"
python main.py --task classification --model DT --kfold 10 && \
echo "Random Forest"
python main.py --task classification --model RF --kfold 10 && \ 
echo "Adaboost"
python main.py --task classification --model ADA --kfold 10 && \ 
echo "Gradient Tree Boosting"
python main.py --task classification --model GT --kfold 10 && \
echo "Support Vector Machine"
python main.py --task classification --model SVM --kfold 10 && \
echo "K-Nearest Neighbor"
python main.py --task classification --model KNN --kfold 10  