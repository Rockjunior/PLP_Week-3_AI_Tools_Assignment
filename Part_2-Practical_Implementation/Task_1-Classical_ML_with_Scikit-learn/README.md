# Classical ML with Scikit-learn: Iris Species Classification

This directory contains a complete machine learning implementation using the classic Iris dataset to demonstrate fundamental ML concepts with Scikit-learn.

## 📁 Files

- `classifying_iris.ipynb` - Main Jupyter notebook with complete ML workflow
- `Dataset/Iris.csv` - The Iris dataset containing flower measurements
- `README.md` - This documentation file

## 🎯 Project Overview

This project implements a classical machine learning pipeline to classify iris flower species based on their physical measurements. The notebook serves as a comprehensive example of best practices in machine learning workflow using Scikit-learn.

### Dataset
The Iris dataset contains 150 samples of iris flowers from three species:
- **Iris-setosa**
- **Iris-versicolor** 
- **Iris-virginica**

Each sample includes four features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## 🔬 Technical Implementation

### Machine Learning Pipeline
The notebook follows a complete ML workflow:

1. **Data Loading & Exploration**
   - Load dataset from CSV
   - Examine data structure and statistics
   - Check for missing values
   - Analyze class distribution

2. **Data Preprocessing**
   - Handle missing values (validation)
   - Encode categorical labels using LabelEncoder
   - Prepare feature matrix and target vector

3. **Model Training**
   - Split data into training (80%) and testing (20%) sets
   - Use stratified sampling to maintain class balance
   - Train Decision Tree classifier with optimized parameters
   - Configure hyperparameters to prevent overfitting

4. **Model Evaluation**
   - Calculate accuracy, precision, and recall metrics
   - Generate detailed classification report
   - Create confusion matrix analysis
   - Compare training vs testing performance

5. **Visualization & Analysis**
   - Confusion matrix heatmap
   - Feature importance visualization
   - Data distribution plots
   - Decision tree structure visualization
   - Performance comparison charts

## 📊 Key Results

The implemented Decision Tree classifier achieves:
- **High accuracy** on the test set
- **Balanced performance** across all three species
- **Interpretable decision rules** for classification
- **Good generalization** without overfitting

## 🛠 Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Basic plotting
- **Seaborn** - Statistical visualization

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Notebook
1. Clone the repository
2. Navigate to this directory
3. Start Jupyter Notebook:
   ```bash
   jupyter notebook classifying_iris.ipynb
   ```
4. Run cells sequentially to see the complete analysis

## 📈 Learning Objectives

This notebook demonstrates:
- **Data preprocessing** techniques for ML
- **Supervised learning** with classification algorithms
- **Model evaluation** using multiple metrics
- **Data visualization** for insights and interpretation
- **Best practices** in ML workflow
- **Decision tree** algorithm implementation and tuning

## 🔍 Key Features

- **Comprehensive Comments**: Every step is thoroughly explained
- **Professional Structure**: Well-organized with clear sections
- **Multiple Visualizations**: Charts and plots for better understanding
- **Performance Analysis**: Detailed evaluation with multiple metrics
- **Reproducible Results**: Fixed random seeds for consistency
- **Best Practices**: Proper data splitting and validation techniques

## 📝 Code Quality

The notebook follows best practices:
- Clear variable naming conventions
- Detailed docstrings and comments
- Modular code structure
- Error handling and validation
- Comprehensive output formatting

## 🎓 Educational Value

Perfect for:
- Learning classical machine learning concepts
- Understanding the Scikit-learn library
- Practicing data preprocessing techniques
- Exploring data visualization methods
- Understanding model evaluation metrics

## 📋 Assignment Requirements Fulfilled

✅ **Preprocess the data** (handle missing values, encode labels)  
✅ **Train a decision tree classifier** to predict iris species  
✅ **Evaluate using accuracy, precision, and recall**  
✅ **Jupyter notebook with comments** explaining each step  

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different algorithms
- Add more visualization techniques
- Implement cross-validation
- Try feature engineering approaches

## 📄 License

This project is for educational purposes as part of the PLP Week 3 AI Tools Assignment.
