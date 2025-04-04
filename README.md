# Laptop Price Prediction

This project aims to predict laptop prices using machine learning regression models. It involves data preprocessing, feature engineering, model training, and evaluation.

## Project Structure

- `laptopPrice.csv`: Dataset containing laptop specifications and prices.
- `laptop_pred.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.

## Logic and Algorithms

1. **Data Preprocessing:**
    - Load the dataset using pandas.
    - Handle missing values (if any).
    - Convert categorical features into numerical using Label Encoding.
    - Split the data into training and testing sets.
    - Scale numerical features using StandardScaler.
2. **Model Training and Evaluation:**
    - Train multiple regression models:
        - Gradient Boosting Regressor
        - AdaBoost Regressor
        - XGBoost Regressor
        - LightGBM Regressor
        - Stacking Regressor (ensemble of the above models)
    - Evaluate model performance using Mean Squared Error (MSE) and R-squared (R2) score.
3. **Visualization:**
    - Create visualizations to compare the performance of different models, including scatter plots and line plots of actual vs. predicted prices.

## Technologies Used

- **Python:** Programming language used for data analysis and machine learning.
- **Pandas:** Library for data manipulation and analysis.
- **Scikit-learn:** Library for machine learning algorithms and model evaluation.
- **XGBoost:** Gradient boosting library for efficient and scalable model training.
- **LightGBM:** Gradient boosting library with high performance and efficiency.
- **Matplotlib and Seaborn:** Libraries for data visualization.

![image](https://github.com/user-attachments/assets/811b1c50-cb6b-4530-9d76-84538671c409)
![image](https://github.com/user-attachments/assets/1e811e9a-3e4c-49cf-a245-1e31be854f17)


## Algorithms Explained

- **Gradient Boosting:** An ensemble method that combines weak learners (decision trees) to create a strong predictive model.
- **AdaBoost:** Another ensemble method that focuses on improving the performance of weak learners by assigning weights to data points.
- **XGBoost:** An optimized gradient boosting library that is widely used for its speed and accuracy.
- **LightGBM:** A gradient boosting library that uses histogram-based algorithms for faster training and lower memory usage.
- **Stacking:** An ensemble method that combines the predictions of multiple models to create a more robust prediction.

## Conclusion

This project demonstrates the application of machine learning regression models to predict laptop prices. The results show that the ensemble models, particularly the Stacking Regressor, achieved the best performance in terms of MSE and R2 score. The visualizations provide insights into the model predictions and their comparison with actual prices.

## Future Work

- Explore other feature engineering techniques to improve model accuracy.
- Fine-tune the hyperparameters of the models for better performance.
- Deploy the model for real-time price prediction.
