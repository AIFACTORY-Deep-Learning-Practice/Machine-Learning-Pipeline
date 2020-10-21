echo "Logistic Regression"
python main.py --task classification --model Logistic --val_size 0.3 && \
echo "Ridge Regression"
python main.py --task classification --model Ridge --val_size 0.3 && \
echo "Decision Tree"
python main.py --task classification --model DT --val_size 0.3 && \
echo "Random Forest"
python main.py --task classification --model RF --val_size 0.3 && \ 
echo "Adaboost"
python main.py --task classification --model ADA --val_size 0.3 && \ 
echo "Gradient Tree Boosting"
python main.py --task classification --model GT --val_size 0.3 && \
echo "Support Vector Machine"
python main.py --task classification --model SVM --val_size 0.3 && \
echo "K-Nearest Neighbor"
python main.py --task classification --model KNN --val_size 0.3  