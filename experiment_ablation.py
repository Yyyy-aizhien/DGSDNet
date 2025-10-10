#!/usr/bin/env python3
"""
Experiment 2: Ablation Studies with FULL DGSDNet
Test: w/o SP, w/o TP, w/o DSD, w/o GSC
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime

# Add project path
sys.path.append('.')

# Import from experiment_missing_rate
from experiment_missing_rate import train_missing_rate_experiment

# Create results directory if not exists
os.makedirs('results', exist_ok=True)

def run_ablation_experiments():
    """Run all ablation experiments"""
    datasets = ['iemocap', 'cmumosi']
    ablation_types = ['full', 'w/o_SP', 'w/o_TP', 'w/o_DSD', 'w/o_GSC']
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = {}
    
    print("\n" + "="*80)
    print("Batch Ablation Experiments [FULL DGSDNet Model]")
    print("="*80)
    print(f"Datasets: {', '.join([d.upper() for d in datasets])}")
    print(f"Ablation Types: {', '.join(ablation_types)}")
    print(f"Testing across all missing rates: {', '.join([f'{mr*100:.0f}%' for mr in missing_rates])}")
    print(f"Total: {len(datasets)} x {len(ablation_types)} x {len(missing_rates)} experiments")
    print("="*80 + "\n")
    
    experiment_count = 0
    total_experiments = len(datasets) * len(ablation_types) * len(missing_rates)
    
    for dataset in datasets:
        results[dataset] = {}
        
        for ablation_type in ablation_types:
            results[dataset][ablation_type] = {}
            
            print(f"\n{'='*70}")
            print(f"Testing: {dataset.upper()} - {ablation_type}")
            print(f"{'='*70}")
            
            waf1_scores = []
            
            for missing_rate in missing_rates:
                experiment_count += 1
                print(f"\nProgress: {experiment_count}/{total_experiments} | Missing Rate: {missing_rate*100:.0f}%")
                
                waf1 = train_missing_rate_experiment(
                    dataset_name=dataset,
                    missing_rate=missing_rate,
                    cv_fold=0,
                    epochs=100,
                    ablation_type=ablation_type
                )
                
                results[dataset][ablation_type][missing_rate] = waf1
                waf1_scores.append(waf1)
            
            # Calculate average WAF1 across all missing rates
            avg_waf1 = np.mean(waf1_scores)
            results[dataset][ablation_type]['average'] = avg_waf1
            
            print(f"\nAverage WAF1 for {ablation_type}: {avg_waf1*100:.2f}%")
    
    # Save results to results/ directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'results/ablation_table2_{timestamp}.json'
    
    save_data = {
        'experiment': 'TABLE 2 - Ablation Studies',
        'timestamp': timestamp,
        'results': results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Print results table (matching paper format - Table 2)
    print(f"\n{'='*80}")
    print("Ablation Studies Results")
    print("AWF1 scores averaged over missing rates 0.0-0.7")
    print(f"{'='*80}\n")
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        print(f"{'Model':<20}{'AWF1 (%)':<12}{'Reduction'}")
        print("-" * 50)
        
        full_waf1 = results[dataset]['full']['average']
        print(f"{'Our method':<20}{full_waf1*100:<12.2f}{'-'}")
        print("-" * 50)
        
        for ablation_type in ['w/o_SP', 'w/o_TP', 'w/o_DSD', 'w/o_GSC']:
            waf1 = results[dataset][ablation_type]['average']
            reduction = full_waf1 - waf1
            print(f"{ablation_type:<20}{waf1*100:<12.2f}(-{reduction*100:.2f})")
        print()
    
    # Print detailed results per missing rate
    print(f"\n{'='*80}")
    print("Detailed Results by Missing Rate")
    print(f"{'='*80}\n")
    
    for dataset in datasets:
        print(f"\n{dataset.upper()} - Detailed Results:")
        print(f"{'Model':<15}", end='')
        for mr in missing_rates:
            print(f"{mr*100:>6.0f}%", end='')
        print(f"{'Avg':>8}")
        print("-" * 80)
        
        for ablation_type in ablation_types:
            print(f"{ablation_type:<15}", end='')
            for mr in missing_rates:
                waf1 = results[dataset][ablation_type][mr]
                print(f"{waf1*100:>6.2f}", end='')
            avg = results[dataset][ablation_type]['average']
            print(f"{avg*100:>8.2f}")
        print()
    
    print(f"{'='*80}")
    print(f"All Experiments Completed!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ablation Experiments with Full DGSDNet')
    parser.add_argument('--dataset', type=str, default='iemocap', choices=['iemocap', 'cmumosi'],
                       help='Dataset selection')
    parser.add_argument('--ablation_type', type=str, default='full',
                       choices=['full', 'w/o_SP', 'w/o_TP', 'w/o_DSD', 'w/o_GSC'],
                       help='Ablation type')
    parser.add_argument('--missing_rate', type=float, default=0.0,
                       help='Missing rate for single experiment')
    parser.add_argument('--cv_fold', type=int, default=0,
                       help='Cross-validation fold')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100 for paper results)')
    parser.add_argument('--run_all', action='store_true',
                       help='Run all ablation experiments across all missing rates')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_ablation_experiments()
    else:
        # Run single ablation experiment
        from experiment_missing_rate import train_missing_rate_experiment
        
        waf1 = train_missing_rate_experiment(
            dataset_name=args.dataset,
            missing_rate=args.missing_rate,
            cv_fold=args.cv_fold,
            epochs=args.epochs,
            ablation_type=args.ablation_type
        )
        print(f"\nFinal WAF1: {waf1*100:.2f}%")