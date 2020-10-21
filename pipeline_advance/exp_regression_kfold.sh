echo "Ordinary Least Square"
python main.py --task regression --model OLS --kfold 10 && \
echo "Ridge Regression"
python main.py --task regression --model Ridge --kfold 10 && \
echo "Lasso Regression"
python main.py --task regression --model Lasso --kfold 10 && \
echo "Elastic-Net"
python main.py --task regression --model ElasticNet --kfold 10 && \
echo "Decision Tree"
python main.py --task regression --model DT --kfold 10 && \
echo "Random Forest"
python main.py --task regression --model RF --kfold 10 && \ 
echo "Adaboost"
python main.py --task regression --model ADA --kfold 10 && \ 
echo "Gradient Tree Boosting"
python main.py --task regression --model GT --kfold 10 && \
echo "Support Vector Machine"
python main.py --task regression --model SVM --kfold 10 && \
echo "K-Nearest Neighbor"
python main.py --task regression --model KNN --kfold 10  