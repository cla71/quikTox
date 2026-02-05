"""
Comprehensive ML Model Assessment Plan for Safety Target Prediction
====================================================================

Goal: Determine the best-fit model for predicting binding/inhibition probability
      with uncertainty quantification for safety pharmacology targets

Author: Christian L. Andersen
Date: 2026-02-04
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate,
    learning_curve, validation_curve
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    calibration_curve, brier_score_loss, log_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ModelAssessmentPlan:
    """
    Comprehensive framework for assessing ML models for safety prediction
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}
        
    # =====================================================================
    # PHASE 1: DATA PREPARATION AND FEATURE ENGINEERING
    # =====================================================================
    
    @staticmethod
    def compute_molecular_descriptors(smiles_list: List[str]) -> pd.DataFrame:
        """
        Compute molecular descriptors using RDKit
        
        Key descriptor families:
        - Constitutional: MW, heavy atom count, rotatable bonds
        - Topological: connectivity indices, shape indices
        - Electronic: partial charges, dipole moment
        - Geometric: 3D properties, surface area
        - Pharmacophore: HBA, HBD, PSA
        - Lipophilicity: LogP, MR
        
        Returns:
        --------
        pd.DataFrame : Descriptor matrix (n_compounds x n_descriptors)
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf
        
        descriptor_functions = {
            'MW': Descriptors.MolWt,
            'LogP': Crippen.MolLogP,
            'HBA': Lipinski.NumHAcceptors,
            'HBD': Lipinski.NumHDonors,
            'TPSA': MolSurf.TPSA,
            'RotatableBonds': Lipinski.NumRotatableBonds,
            'AromaticRings': Lipinski.NumAromaticRings,
            'HeavyAtoms': Lipinski.HeavyAtomCount,
            'FractionCSP3': Lipinski.FractionCsp3,
            'MolMR': Crippen.MolMR,
            # Add more as needed
        }
        
        descriptors = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc_dict = {name: func(mol) for name, func in descriptor_functions.items()}
                descriptors.append(desc_dict)
            else:
                descriptors.append({name: np.nan for name in descriptor_functions.keys()})
        
        return pd.DataFrame(descriptors)
    
    @staticmethod
    def compute_molecular_fingerprints(smiles_list: List[str], 
                                      fp_type: str = 'morgan',
                                      n_bits: int = 2048) -> np.ndarray:
        """
        Compute molecular fingerprints
        
        Fingerprint types:
        - Morgan (ECFP): Circular fingerprints, radius 2-3
        - MACCS: 166-bit structural keys
        - Topological: Path-based fingerprints
        - Atom Pair: Considers pairs of atoms
        - Topological Torsion: 4-atom fragments
        - RDKit: Daylight-like fingerprints
        
        Returns:
        --------
        np.ndarray : Binary fingerprint matrix (n_compounds x n_bits)
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys
        
        fingerprints = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                if fp_type == 'morgan':
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                elif fp_type == 'maccs':
                    fp = MACCSkeys.GenMACCSKeys(mol)
                elif fp_type == 'topological':
                    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
                else:
                    raise ValueError(f"Unknown fingerprint type: {fp_type}")
                
                fingerprints.append(np.array(fp))
            else:
                fingerprints.append(np.zeros(n_bits))
        
        return np.array(fingerprints)
    
    @staticmethod
    def feature_selection_and_engineering(X: pd.DataFrame, 
                                         y: np.ndarray,
                                         method: str = 'variance') -> pd.DataFrame:
        """
        Feature selection to reduce dimensionality and improve model performance
        
        Methods:
        - Variance threshold: Remove low-variance features
        - Correlation filter: Remove highly correlated features
        - Recursive feature elimination (RFE)
        - L1 regularization (Lasso)
        - Tree-based importance
        - Mutual information
        
        Returns:
        --------
        pd.DataFrame : Selected feature matrix
        """
        from sklearn.feature_selection import (
            VarianceThreshold, SelectKBest, mutual_info_classif,
            RFE, SelectFromModel
        )
        from sklearn.linear_model import LogisticRegression
        
        if method == 'variance':
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            return pd.DataFrame(X_selected, columns=X.columns[selector.get_support()])
        
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(100, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            return pd.DataFrame(X_selected, 
                              columns=X.columns[selector.get_support()])
        
        elif method == 'l1':
            selector = SelectFromModel(
                LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
            )
            X_selected = selector.fit_transform(X, y)
            return pd.DataFrame(X_selected, 
                              columns=X.columns[selector.get_support()])
        
        return X
    
    # =====================================================================
    # PHASE 2: MODEL TRAINING AND HYPERPARAMETER OPTIMIZATION
    # =====================================================================
    
    def get_model_configurations(self) -> Dict[str, Dict]:
        """
        Define model configurations with hyperparameter search spaces
        
        Models to evaluate:
        1. Random Forest (RF)
        2. Gradient Boosting (XGBoost, LightGBM, CatBoost)
        3. Neural Networks (MLP, Deep Learning)
        4. Support Vector Machines (SVM)
        5. Gaussian Processes
        6. Ensemble methods
        
        Returns:
        --------
        Dict : Model configurations with hyperparameter grids
        """
        
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.3],
                    'class_weight': ['balanced', None]
                },
                'requires_scaling': False
            },
            
            'XGBoost': {
                'model': None,  # Will be imported dynamically
                'params': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.5],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 2, 5]
                },
                'requires_scaling': False
            },
            
            'LightGBM': {
                'model': None,
                'params': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 63, 127],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 2, 5]
                },
                'requires_scaling': False
            },
            
            'NeuralNetwork_MLP': {
                'model': None,  # MLPClassifier
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [500, 1000]
                },
                'requires_scaling': True
            },
            
            'SVM_RBF': {
                'model': None,  # SVC with probability=True
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'class_weight': ['balanced', None]
                },
                'requires_scaling': True
            }
        }
        
        return models
    
    def hyperparameter_optimization(self, 
                                   X: np.ndarray, 
                                   y: np.ndarray,
                                   model_name: str,
                                   optimization_method: str = 'random_search',
                                   n_iter: int = 50,
                                   cv: int = 5) -> Dict:
        """
        Optimize hyperparameters using various search strategies
        
        Methods:
        - Grid Search: Exhaustive search over parameter grid
        - Random Search: Random sampling from parameter distributions
        - Bayesian Optimization: Smart search using Gaussian processes
        - Hyperband: Resource-efficient successive halving
        - Optuna: Tree-structured Parzen estimator
        
        Returns:
        --------
        Dict : Best parameters and cross-validation scores
        """
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        
        models = self.get_model_configurations()
        model_config = models[model_name]
        
        if optimization_method == 'random_search':
            search = RandomizedSearchCV(
                estimator=model_config['model'],
                param_distributions=model_config['params'],
                n_iter=n_iter,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        elif optimization_method == 'grid_search':
            search = GridSearchCV(
                estimator=model_config['model'],
                param_grid=model_config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        search.fit(X, y)
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
    
    # =====================================================================
    # PHASE 3: MODEL EVALUATION AND VALIDATION
    # =====================================================================
    
    def stratified_cross_validation(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   model,
                                   n_splits: int = 5,
                                   n_repeats: int = 3) -> Dict:
        """
        Rigorous cross-validation with multiple repeats
        
        Evaluation metrics:
        - ROC-AUC: Overall discrimination ability
        - PR-AUC: Performance on imbalanced datasets
        - Sensitivity/Specificity: At various thresholds
        - MCC: Matthews correlation coefficient (balanced measure)
        - Cohen's Kappa: Inter-rater reliability
        - Calibration metrics: Brier score, log loss
        
        Returns:
        --------
        Dict : Cross-validation results with confidence intervals
        """
        from sklearn.model_selection import RepeatedStratifiedKFold
        
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, 
            n_repeats=n_repeats,
            random_state=self.random_state
        )
        
        scoring = {
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'accuracy': 'accuracy'
        }
        
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate confidence intervals
        results_summary = {}
        for metric, scores in cv_results.items():
            if 'test_' in metric:
                metric_name = metric.replace('test_', '')
                results_summary[metric_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'ci_lower': np.percentile(scores, 2.5),
                    'ci_upper': np.percentile(scores, 97.5),
                    'scores': scores
                }
        
        return results_summary
    
    def external_validation(self,
                           model,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray) -> Dict:
        """
        External validation on held-out test set
        
        Key assessments:
        1. Generalization performance
        2. Overfitting detection
        3. Calibration quality
        4. Decision boundary analysis
        5. Error analysis
        
        Returns:
        --------
        Dict : Comprehensive evaluation metrics
        """
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        results = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Calculate calibration metrics
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        
        results['calibration'] = {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
        
        return results
    
    def applicability_domain_analysis(self,
                                     X_train: np.ndarray,
                                     X_test: np.ndarray,
                                     method: str = 'leverage') -> Dict:
        """
        Assess applicability domain - reliability of predictions
        
        Methods:
        - Leverage approach: Mahalanobis distance
        - Distance to model: k-NN distance
        - Probability density: Local outlier factor
        - Ensemble standard deviation
        
        Returns predictions should be flagged as "outside domain" when:
        - Structural dissimilarity to training set
        - Extreme descriptor values
        - High prediction uncertainty
        
        Returns:
        --------
        Dict : Applicability domain metrics and flags
        """
        from scipy.spatial.distance import cdist
        from sklearn.neighbors import LocalOutlierFactor
        
        if method == 'leverage':
            # Calculate leverage (hat values)
            from scipy.linalg import svd
            U, s, Vh = svd(X_train, full_matrices=False)
            leverage_train = np.sum(U**2, axis=1)
            
            # For test set, use distance to training set centroid
            centroid = np.mean(X_train, axis=0)
            distances = cdist(X_test, centroid.reshape(1, -1), metric='mahalanobis')
            
            warning_threshold = 3 * np.mean(leverage_train)
            
            return {
                'method': 'leverage',
                'distances': distances.flatten(),
                'threshold': warning_threshold,
                'outside_domain': distances.flatten() > warning_threshold
            }
        
        elif method == 'lof':
            # Local outlier factor
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
            lof.fit(X_train)
            
            outlier_scores = lof.score_samples(X_test)
            
            return {
                'method': 'local_outlier_factor',
                'scores': outlier_scores,
                'outside_domain': outlier_scores < -1.5
            }
        
        return {}
    
    # =====================================================================
    # PHASE 4: UNCERTAINTY QUANTIFICATION
    # =====================================================================
    
    def conformal_prediction(self,
                            model,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_calibration: np.ndarray,
                            y_calibration: np.ndarray,
                            X_test: np.ndarray,
                            confidence_level: float = 0.95) -> Dict:
        """
        Conformal prediction for rigorous uncertainty quantification
        
        Provides prediction sets with coverage guarantees:
        - P(y_true ∈ prediction_set) ≥ confidence_level
        
        Types:
        - Inductive conformal: Split conformal prediction
        - Transductive conformal: Full conformal prediction
        - Mondrian conformal: Class-conditional conformity
        
        Returns:
        --------
        Dict : Prediction sets with coverage guarantees
        """
        
        # Train model on training set
        model.fit(X_train, y_train)
        
        # Calculate nonconformity scores on calibration set
        y_cal_pred_proba = model.predict_proba(X_calibration)
        
        # Nonconformity score: 1 - probability of true class
        nonconformity_scores = []
        for i, y_true in enumerate(y_calibration):
            score = 1 - y_cal_pred_proba[i, int(y_true)]
            nonconformity_scores.append(score)
        
        nonconformity_scores = np.array(nonconformity_scores)
        
        # Calculate quantile threshold
        n = len(nonconformity_scores)
        q = np.ceil((n + 1) * confidence_level) / n
        threshold = np.quantile(nonconformity_scores, q)
        
        # Make predictions on test set
        y_test_pred_proba = model.predict_proba(X_test)
        
        # Generate prediction sets
        prediction_sets = []
        for probs in y_test_pred_proba:
            pred_set = []
            for class_idx, prob in enumerate(probs):
                if (1 - prob) <= threshold:
                    pred_set.append(class_idx)
            prediction_sets.append(pred_set)
        
        return {
            'threshold': threshold,
            'prediction_sets': prediction_sets,
            'set_sizes': [len(s) for s in prediction_sets],
            'confidence_level': confidence_level
        }
    
    def monte_carlo_dropout(self,
                           model,
                           X_test: np.ndarray,
                           n_iterations: int = 100,
                           dropout_rate: float = 0.1) -> Dict:
        """
        Monte Carlo Dropout for neural network uncertainty
        
        Method:
        - Apply dropout at test time
        - Run multiple forward passes
        - Estimate prediction variance
        
        Useful for:
        - Epistemic uncertainty (model uncertainty)
        - Active learning (select uncertain samples)
        - Reliability assessment
        
        Returns:
        --------
        Dict : Prediction means and uncertainties
        """
        
        predictions = []
        
        for _ in range(n_iterations):
            # Note: This requires the model to have dropout layers enabled at test time
            # Implementation depends on framework (TensorFlow, PyTorch, etc.)
            y_pred = model.predict_proba(X_test)[:, 1]
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'ci_lower': np.percentile(predictions, 2.5, axis=0),
            'ci_upper': np.percentile(predictions, 97.5, axis=0),
            'predictions': predictions
        }
    
    def ensemble_uncertainty(self,
                            models: List,
                            X_test: np.ndarray) -> Dict:
        """
        Ensemble-based uncertainty quantification
        
        Approaches:
        - Bootstrap aggregating (bagging)
        - Different model architectures
        - Different hyperparameters
        - Different feature subsets
        
        Uncertainty decomposition:
        - Epistemic: Model uncertainty (variance across models)
        - Aleatoric: Data uncertainty (inherent noise)
        
        Returns:
        --------
        Dict : Ensemble predictions with uncertainty estimates
        """
        
        predictions = []
        
        for model in models:
            y_pred = model.predict_proba(X_test)[:, 1]
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        # Calculate disagreement metrics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Coefficient of variation
        cv = std_pred / (mean_pred + 1e-10)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'cv': cv,
            'ci_lower': np.percentile(predictions, 2.5, axis=0),
            'ci_upper': np.percentile(predictions, 97.5, axis=0),
            'n_models': len(models)
        }
    
    # =====================================================================
    # PHASE 5: MODEL COMPARISON AND SELECTION
    # =====================================================================
    
    def statistical_comparison(self,
                              model_results: Dict[str, Dict],
                              metric: str = 'roc_auc',
                              alpha: float = 0.05) -> pd.DataFrame:
        """
        Statistical comparison of model performance
        
        Tests:
        - Paired t-test: Compare two models on same CV folds
        - ANOVA: Compare multiple models
        - Friedman test: Non-parametric alternative to ANOVA
        - Wilcoxon signed-rank: Non-parametric paired comparison
        - Bonferroni correction: Multiple comparison adjustment
        
        Returns:
        --------
        pd.DataFrame : Statistical test results with p-values
        """
        from scipy import stats
        
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        # Create comparison matrix
        comparison_matrix = np.zeros((n_models, n_models))
        p_values = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    scores1 = model_results[model1][metric]['scores']
                    scores2 = model_results[model2][metric]['scores']
                    
                    # Paired t-test
                    t_stat, p_val = stats.ttest_rel(scores1, scores2)
                    
                    comparison_matrix[i, j] = np.mean(scores1) - np.mean(scores2)
                    p_values[i, j] = p_val
        
        # Apply Bonferroni correction
        n_comparisons = n_models * (n_models - 1) / 2
        bonferroni_alpha = alpha / n_comparisons
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Model 1': [],
            'Model 2': [],
            'Mean Difference': [],
            'p-value': [],
            'Significant (Bonferroni)': []
        })
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                results = pd.concat([results, pd.DataFrame({
                    'Model 1': [model_names[i]],
                    'Model 2': [model_names[j]],
                    'Mean Difference': [comparison_matrix[i, j]],
                    'p-value': [p_values[i, j]],
                    'Significant (Bonferroni)': [p_values[i, j] < bonferroni_alpha]
                })], ignore_index=True)
        
        return results.sort_values('p-value')
    
    def model_selection_criteria(self,
                                model_results: Dict[str, Dict],
                                weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Multi-criteria decision analysis for model selection
        
        Criteria:
        1. Predictive performance (ROC-AUC, PR-AUC, MCC)
        2. Calibration quality (Brier score, calibration slope)
        3. Computational efficiency (training time, inference time)
        4. Interpretability (feature importance, SHAP values)
        5. Robustness (performance variance, stability)
        6. Uncertainty quality (calibration, sharpness)
        
        Weighting schemes:
        - Equal weights: All criteria equal importance
        - Custom weights: Domain-specific priorities
        - Analytic Hierarchy Process (AHP)
        
        Returns:
        --------
        pd.DataFrame : Ranked models with composite scores
        """
        
        if weights is None:
            weights = {
                'roc_auc': 0.25,
                'pr_auc': 0.20,
                'calibration': 0.20,
                'robustness': 0.15,
                'efficiency': 0.10,
                'interpretability': 0.10
            }
        
        model_scores = []
        
        for model_name, results in model_results.items():
            # Normalize scores to 0-1 scale
            score_dict = {
                'model': model_name,
                'roc_auc_score': results['roc_auc']['mean'],
                'pr_auc_score': results.get('average_precision', {}).get('mean', 0),
                'robustness_score': 1 - results['roc_auc']['std'],  # Lower variance = higher score
                # Add other normalized scores
            }
            
            # Calculate weighted composite score
            composite_score = sum(
                score_dict.get(f"{k}_score", 0) * v 
                for k, v in weights.items()
            )
            
            score_dict['composite_score'] = composite_score
            model_scores.append(score_dict)
        
        df = pd.DataFrame(model_scores)
        return df.sort_values('composite_score', ascending=False)
    
    # =====================================================================
    # PHASE 6: REPORTING AND VISUALIZATION
    # =====================================================================
    
    def generate_performance_plots(self,
                                  model_results: Dict,
                                  y_true: np.ndarray,
                                  y_pred_proba: Dict[str, np.ndarray],
                                  save_dir: str = '.') -> None:
        """
        Generate comprehensive performance visualization plots
        
        Plots:
        1. ROC curves with confidence intervals
        2. Precision-Recall curves
        3. Calibration curves
        4. Learning curves
        5. Feature importance plots
        6. Confusion matrices
        7. Error analysis plots
        8. Uncertainty visualization
        
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # ROC curves
        ax = axes[0, 0]
        for model_name, y_pred in y_pred_proba.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision-Recall curves
        ax = axes[0, 1]
        for model_name, y_pred in y_pred_proba.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
            ax.plot(recall, precision, label=f'{model_name} (PR-AUC={pr_auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calibration curves
        ax = axes[0, 2]
        for model_name, y_pred in y_pred_proba.items():
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=10
            )
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   marker='o', label=model_name)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Model comparison boxplot
        ax = axes[1, 0]
        comparison_data = []
        labels = []
        for model_name, results in model_results.items():
            comparison_data.append(results['roc_auc']['scores'])
            labels.append(model_name)
        ax.boxplot(comparison_data, labels=labels)
        ax.set_ylabel('ROC-AUC')
        ax.set_title('Model Performance Comparison')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/model_performance_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_final_report(self,
                            model_results: Dict,
                            best_model: str,
                            save_path: str = 'model_assessment_report.txt') -> None:
        """
        Generate comprehensive text report
        
        Report sections:
        1. Executive summary
        2. Data characteristics
        3. Model performance comparison
        4. Statistical significance tests
        5. Uncertainty quantification results
        6. Applicability domain analysis
        7. Recommendations
        8. Limitations and caveats
        
        """
        
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ML MODEL ASSESSMENT REPORT\n")
            f.write("Safety Pharmacology Prediction Models\n")
            f.write("="*80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Best performing model: {best_model}\n")
            f.write(f"Total models evaluated: {len(model_results)}\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-"*80 + "\n")
            for model_name, results in model_results.items():
                f.write(f"\n{model_name}:\n")
                for metric, values in results.items():
                    f.write(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}\n")
                    f.write(f"    95% CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")


# =====================================================================
# MAIN EXECUTION PLAN
# =====================================================================

def main_assessment_workflow():
    """
    Main workflow for comprehensive model assessment
    
    Workflow steps:
    1. Load and preprocess data
    2. Feature engineering (descriptors + fingerprints)
    3. Train multiple model types
    4. Hyperparameter optimization
    5. Cross-validation evaluation
    6. External validation
    7. Uncertainty quantification
    8. Statistical comparison
    9. Model selection
    10. Generate reports and visualizations
    """
    
    print("="*80)
    print("COMPREHENSIVE ML MODEL ASSESSMENT WORKFLOW")
    print("="*80)
    
    # Initialize assessment framework
    assessment = ModelAssessmentPlan(random_state=42)
    
    # Phase 1: Data preparation
    print("\nPhase 1: Data Preparation and Feature Engineering")
    print("-"*80)
    print("Tasks:")
    print("  1. Load bioactivity data from ChEMBL")
    print("  2. Clean and preprocess (remove duplicates, handle missing values)")
    print("  3. Generate molecular descriptors (RDKit)")
    print("  4. Generate molecular fingerprints (Morgan, MACCS, etc.)")
    print("  5. Feature selection and dimensionality reduction")
    print("  6. Train/validation/test split (60/20/20)")
    
    # Phase 2: Model training
    print("\nPhase 2: Model Training and Hyperparameter Optimization")
    print("-"*80)
    print("Models to evaluate:")
    models = assessment.get_model_configurations()
    for i, (model_name, config) in enumerate(models.items(), 1):
        print(f"  {i}. {model_name}")
    print("\nOptimization strategy:")
    print("  - Random search with 50 iterations")
    print("  - 5-fold cross-validation for parameter selection")
    print("  - Optimize for ROC-AUC")
    
    # Phase 3: Evaluation
    print("\nPhase 3: Model Evaluation and Validation")
    print("-"*80)
    print("Evaluation protocol:")
    print("  - Stratified 5-fold CV with 3 repeats (15 total evaluations)")
    print("  - Metrics: ROC-AUC, PR-AUC, MCC, Sensitivity, Specificity")
    print("  - External validation on held-out test set")
    print("  - Applicability domain assessment")
    
    # Phase 4: Uncertainty quantification
    print("\nPhase 4: Uncertainty Quantification")
    print("-"*80)
    print("Methods:")
    print("  1. Conformal prediction (split conformal)")
    print("  2. Monte Carlo dropout (for neural networks)")
    print("  3. Ensemble uncertainty (bootstrap aggregating)")
    print("  4. Calibration assessment (reliability diagrams)")
    
    # Phase 5: Model comparison
    print("\nPhase 5: Statistical Model Comparison")
    print("-"*80)
    print("Statistical tests:")
    print("  - Paired t-tests with Bonferroni correction")
    print("  - Multi-criteria decision analysis")
    print("  - Effect size calculations (Cohen's d)")
    
    # Phase 6: Reporting
    print("\nPhase 6: Reporting and Visualization")
    print("-"*80)
    print("Deliverables:")
    print("  1. Comprehensive performance plots")
    print("  2. Statistical comparison tables")
    print("  3. Detailed text report with recommendations")
    print("  4. Model cards for deployment")
    
    print("\n" + "="*80)
    print("Assessment workflow plan complete.")
    print("="*80)


if __name__ == "__main__":
    main_assessment_workflow()
