echo "Ordinary Least Square"
python main.py --task regression --model OLS --val_size 0.3 && \
echo "Ridge Regression"
python main.py --task regression --model Ridge --val_size 0.3 && \
echo "Lasso Regression"
python main.py --task regression --model Lasso --val_size 0.3 && \
echo "Elastic-Net"
python main.py --task regression --model ElasticNet --val_size 0.3 && \
echo "Decision Tree"
python main.py --task regression --model DT --val_size 0.3 && \
echo "Random Forest"
python main.py --task regression --model RF --val_size 0.3 && \ 
echo "Adaboost"
python main.py --task regression --model ADA --val_size 0.3 && \ 
echo "Gradient Tree Boosting"
python main.py --task regression --model GT --val_size 0.3 && \
echo "Support Vector Machine"
python main.py --task regression --model SVM --val_size 0.3 && \
echo "K-Nearest Neighbor"
python main.py --task regression --model KNN --val_size 0.3  