#!/usr/bin/env python3
"""
Resume Script for Paper Evaluation Pipeline
Can resume from any point in the pipeline
"""

import argparse
import subprocess
import os
import pandas as pd
from datetime import datetime

class PaperEvaluationResume:
    def __init__(self):
        self.base_dir = "evaluation_pipeline"
        self.experiments_dir = "paper_experiments"
        self.results_csv = "comprehensive_paper_results.csv"
    
    def check_existing_progress(self):
        """Check what has already been completed"""
        print("Checking existing progress...")
        
        completed = {
            'synthetic': {},
            'training': {},
            'evaluation': {}
        }
        
        # Check synthetic data
        synthetic_dir = os.path.join(self.experiments_dir, "synthetic")
        if os.path.exists(synthetic_dir):
            for model_type in ['pretrained', 'from_scratch']:
                model_dir = os.path.join(synthetic_dir, model_type)
                if os.path.exists(model_dir):
                    completed['synthetic'][model_type] = []
                    for method in os.listdir(model_dir):
                        method_dir = os.path.join(model_dir, method)
                        if os.path.exists(os.path.join(method_dir, "imagesTr")) and \
                           os.path.exists(os.path.join(method_dir, "labelsTr")):
                            completed['synthetic'][model_type].append(method)
        
        # Check training results
        training_dir = os.path.join(self.experiments_dir, "training")
        if os.path.exists(training_dir):
            for seg_model in ['nnunet', 'swinunetr']:
                seg_dir = os.path.join(training_dir, seg_model)
                if os.path.exists(seg_dir):
                    completed['training'][seg_model] = []
                    for exp_dir in os.listdir(seg_dir):
                        if os.path.exists(os.path.join(seg_dir, exp_dir, "best_metric_model.pth")):
                            completed['training'][seg_model].append(exp_dir)
        
        # Check evaluation results
        if os.path.exists(self.results_csv):
            df = pd.read_csv(self.results_csv)
            completed['evaluation'] = df['Experiment'].unique().tolist()
        
        return completed
    
    def resume_synthetic_generation(self, method, model_type):
        """Resume synthetic data generation"""
        print(f"\n{'='*60}")
        print(f"RESUMING SYNTHETIC GENERATION: {method.upper()} ({model_type.upper()})")
        print(f"{'='*60}")
        
        output_dir = os.path.join(self.experiments_dir, f"synthetic/{model_type}/{method}")
        
        if os.path.exists(output_dir) and \
           os.path.exists(os.path.join(output_dir, "imagesTr")) and \
           os.path.exists(os.path.join(output_dir, "labelsTr")):
            print(f"✓ Synthetic data already exists for {method} ({model_type})")
            return True
        
        # Import the main pipeline class
        from run_comprehensive_paper_evaluation import PaperEvaluationPipeline
        pipeline = PaperEvaluationPipeline()
        
        if model_type == "pretrained":
            return pipeline.generate_synthetic_data_pretrained(method)
        else:
            return pipeline.generate_synthetic_data_from_scratch(method)
    
    def resume_training(self, method, model_type, segmentation_model):
        """Resume model training"""
        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING: {method.upper()} + {segmentation_model.upper()}")
        print(f"Model Type: {model_type}")
        print(f"{'='*60}")
        
        training_output_dir = os.path.join(self.experiments_dir, f"training/{segmentation_model.lower()}/{method}_{model_type}")
        
        if os.path.exists(os.path.join(training_output_dir, "best_metric_model.pth")):
            print(f"✓ Training already completed for {method} + {segmentation_model}")
            return True
        
        # Import the main pipeline class
        from run_comprehensive_paper_evaluation import PaperEvaluationPipeline
        pipeline = PaperEvaluationPipeline()
        
        return pipeline.train_segmentation_model(method, model_type, segmentation_model)
    
    def resume_evaluation(self, method, model_type, segmentation_model):
        """Resume model evaluation"""
        print(f"\n{'='*60}")
        print(f"RESUMING EVALUATION: {method.upper()} + {segmentation_model.upper()}")
        print(f"Model Type: {model_type}")
        print(f"{'='*60}")
        
        experiment_name = f"{method}_{model_type}_{segmentation_model.lower()}"
        
        if os.path.exists(self.results_csv):
            df = pd.read_csv(self.results_csv)
            if experiment_name in df['Experiment'].values:
                print(f"✓ Evaluation already completed for {experiment_name}")
                return True
        
        # Import the main pipeline class
        from run_comprehensive_paper_evaluation import PaperEvaluationPipeline
        pipeline = PaperEvaluationPipeline()
        
        return pipeline.evaluate_model(method, model_type, segmentation_model)
    
    def run_resume_pipeline(self, methods=None, model_types=None, segmentation_models=None):
        """Run resume pipeline from current state"""
        if methods is None:
            methods = ["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"]
        if model_types is None:
            model_types = ["pretrained", "from_scratch"]
        if segmentation_models is None:
            segmentation_models = ["nnUNet", "SwinUNETR"]
        
        print("LeFusion Paper Evaluation Resume Pipeline")
        print("=" * 80)
        print(f"Methods: {methods}")
        print(f"Model Types: {model_types}")
        print(f"Segmentation Models: {segmentation_models}")
        
        # Check existing progress
        completed = self.check_existing_progress()
        
        print("\nExisting Progress:")
        print(f"Synthetic Data: {completed['synthetic']}")
        print(f"Training: {completed['training']}")
        print(f"Evaluation: {completed['evaluation']}")
        
        # Resume from where we left off
        for method in methods:
            for model_type in model_types:
                # Resume synthetic generation if needed
                if method not in completed['synthetic'].get(model_type, []):
                    self.resume_synthetic_generation(method, model_type)
                
                # Resume training and evaluation
                for segmentation_model in segmentation_models:
                    seg_model_lower = segmentation_model.lower()
                    exp_name = f"{method}_{model_type}_{seg_model_lower}"
                    
                    # Resume training if needed
                    if seg_model_lower not in completed['training'] or \
                       exp_name not in completed['training'][seg_model_lower]:
                        self.resume_training(method, model_type, segmentation_model)
                    
                    # Resume evaluation if needed
                    if exp_name not in completed['evaluation']:
                        self.resume_evaluation(method, model_type, segmentation_model)
        
        # Generate final results
        from run_comprehensive_paper_evaluation import PaperEvaluationPipeline
        pipeline = PaperEvaluationPipeline()
        pipeline.generate_paper_results_table()

def main():
    parser = argparse.ArgumentParser(description="Resume paper evaluation pipeline")
    parser.add_argument('--methods', nargs='+', 
                       default=['baseline', 'lefusion', 'lefusion_h', 'lefusion_h_diffmask'],
                       help='Methods to evaluate')
    parser.add_argument('--model_types', nargs='+', 
                       default=['pretrained', 'from_scratch'],
                       help='Model types (pretrained/from_scratch)')
    parser.add_argument('--segmentation_models', nargs='+', 
                       default=['nnUNet', 'SwinUNETR'],
                       help='Segmentation models to evaluate')
    parser.add_argument('--check_only', action='store_true',
                       help='Only check existing progress without resuming')
    args = parser.parse_args()
    
    resume_pipeline = PaperEvaluationResume()
    
    if args.check_only:
        completed = resume_pipeline.check_existing_progress()
        print("\nExisting Progress Summary:")
        print(f"Synthetic Data: {completed['synthetic']}")
        print(f"Training: {completed['training']}")
        print(f"Evaluation: {completed['evaluation']}")
    else:
        resume_pipeline.run_resume_pipeline(
            methods=args.methods,
            model_types=args.model_types,
            segmentation_models=args.segmentation_models
        )

if __name__ == '__main__':
    main() 