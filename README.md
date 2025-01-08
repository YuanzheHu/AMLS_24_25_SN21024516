# AMLS_24_25_SN21024516

This project is designed for solving classification tasks using CNN and SVM models, as well as experimenting with different datasets like BreastMNIST and BloodMNIST.

---

## **Setup Instructions**

### **1. Clone the Repository**
First, clone the repository to your local machine:
```bash
git clone <repository-url>
cd <repository-directory>
```

### **2. Create and Activate Environment**
Use Conda to create and activate the project environment:

1. **Create a new Conda environment**:
   ```bash
   conda create --name mls-i python=3.9
   ```

2. **Activate the Conda environment**:
   ```bash
   conda activate mls-i
   ```

3. **Install dependencies**:
   Install all required dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Run the Program**

To execute the main program and evaluate tasks:
```bash
python main.py
```

---

## **Project Workflow**

1. **Task A**:
   - Trains and evaluates CNN and SVM models using the BreastMNIST dataset.
   - Compares the performance of CNN and SVM using metrics like F1-Score and ROC-AUC.
   - Visualizes predictions and confusion matrices.

2. **Task B**:
   - Trains and evaluates ResNet and ViT models using the BloodMNIST dataset.
   - Compares the performance of ResNet and ViT using metrics like F1-Score.
   - Visualizes predictions and confusion matrices.

---

## **Outputs**

- **Logs**: Training logs and metrics are saved in `A/log` and `B/log` directories.
- **Figures**: Model performance visualizations (e.g., confusion matrices, ROC-AUC) are saved in `A/figure` and `B/figure`.

---

## **Troubleshooting**

### **1. Missing Dependencies**
If you encounter missing dependencies, ensure your environment is activated and reinstall the required packages:
```bash
pip install -r requirements.txt
```

### **2. Environment Issues**
To recreate the environment, remove the existing one and follow the setup instructions again:
```bash
conda remove --name mls-i --all
conda create --name mls-i python=3.9
pip install -r requirements.txt
```

---

## **Contributors**

- **Yanzhe Hu**

---

