import warnings
warnings.filterwarnings("ignore")
import json
from src.evaluation.evaluation import Evaluation
from src.methods.explainability import Explainability
from src.methods.feature_performance_weaknesses import FeaturePerformanceWeaknessAnalyzer
from src.methods.uncertainty import Uncertainty
from src.pipeline_manager import PipelineManager    

class ModelImprover:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)

    def run_pipeline(self, pipelines_to_run):
        
        pipeline_mgr = PipelineManager(self.config)

        pipeline_methods = {
            'baseline': pipeline_mgr.train_baseline_model,
            'data_augmentation': pipeline_mgr.train_model_with_data_augmentation,
            'adversarial_training': pipeline_mgr.train_model_with_adversarial_data,
            # 'augmented_adversarial': pipeline_mgr.train_model_with_augmented_and_adversarial_data,
            # 'smote_augmentation': pipeline_mgr.train_model_with_smote_and_augmentation
        }

        trained_pipelines = {}
        for pipeline_name in pipelines_to_run:
            if pipeline_name in pipeline_methods:
                print(f'Training and evaluating {pipeline_name} pipeline...')
                pipeline, model, all_feature_names, X_train_processed, X_test_processed, X_test, y_train, y_test = pipeline_methods[pipeline_name]()
                eval = Evaluation(pipeline, X_test, y_test, self.config)
                auc = eval.get_auc()
                eval.get_roc()
                print(f'{pipeline_name} model AUC: {auc}')
                trained_pipelines[pipeline_name] = (pipeline, model, all_feature_names, X_train_processed, X_test_processed, X_test, y_train, y_test, auc)

        return trained_pipelines

    def run_analysis(self, trained_pipelines, analyses_to_run):

        analysis_methods = {
            'uncertainty': self.run_uncertainty_analysis,
            'feature_importance': self.run_feature_importance_analysis,
            'feature_performance': self.run_feature_performance_analysis
        }

        for analysis_name in analyses_to_run:
            if analysis_name in analysis_methods:
                for pipeline_name, data in trained_pipelines.items():
                    print(f'Running {analysis_name} analysis on {pipeline_name} pipeline...')
                    analysis_methods[analysis_name](pipeline_name, *data)

    def run_uncertainty_analysis(self, pipeline_name, pipeline, model, all_feature_names, X_train_processed, X_test_processed, X_test, y_train, y_test, auc):
        uncertainty_bs = Uncertainty(model, X_train_processed, y_train.values, X_test_processed)
        uncertainty_bs.baseline_ensemble_monte_carlo()
        print(f'{pipeline_name} uncertainty analysis completed.')

    def run_feature_importance_analysis(self, pipeline_name, pipeline, model, all_feature_names, X_train_processed, X_test_processed, X_test, y_train, y_test, auc):
        explainability = Explainability(self.config, model, X_train_processed, all_feature_names)
        explainability.plot_feature_importance()
        explainability.plot_shap_summary()
        selected_features = explainability.select_features_based_on_shap()
        print(f'{pipeline_name} selected features based on SHAP: {selected_features}')

    def run_feature_performance_analysis(self, pipeline_name, pipeline, model, all_feature_names, X_train_processed, X_test_processed, X_test, y_train, y_test, auc):
        fpwa = FeaturePerformanceWeaknessAnalyzer(self.config, model, X_train_processed, y_train, all_feature_names)
        metric_results = fpwa.analyze_feature_performance()
        if metric_results:
            vulnerable_features = fpwa.plot_metric_drops(metric_results)
            print(f'{pipeline_name} vulnerable features: {vulnerable_features}')
            explainability = Explainability(self.config, model, X_train_processed, all_feature_names)
            explainability.plot_partial_dependence(vulnerable_features)
            explainability.plot_shap_dependence(vulnerable_features)
        else:
            print(f'No significant metric drops found for {pipeline_name}.')



if __name__ == "__main__":
    
    config_path = 'configs/german_credit_config.json'
    # config_path = 'configs/marketing_campaign_config.json'
    
    pipelines_to_run = [
        'baseline',
        'data_augmentation',
        'adversarial_training',
        # 'augmented_adversarial',  # This pipeline is broken
        # 'smote_augmentation' # This pipeline is broken
    ]
    analyses_to_run = [
        'uncertainty',
        'feature_importance',
        'feature_performance'
    ]
    model_improver = ModelImprover(config_path)
    trained_pipelines = model_improver.run_pipeline(pipelines_to_run)
    
    # print(trained_pipelines, type(trained_pipelines))
    trained_pipelines = {'baseline': trained_pipelines['baseline']} # Analysis only on baseline model
    model_improver.run_analysis(trained_pipelines, analyses_to_run)