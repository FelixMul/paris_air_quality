"""
Compare Results: Basic vs Enhanced Features
==========================================

This script compares the performance of:
1. Basic model (5 features)
2. Enhanced model (22 features with feature engineering)
"""

import pandas as pd
import numpy as np

def compare_results():
    """Compare basic vs enhanced model results"""
    
    print("=" * 80)
    print("FEATURE ENGINEERING RESULTS COMPARISON")
    print("=" * 80)
    
    # Results from basic model (5 features)
    basic_results = {
        'valeur_CO': 0.0759,
        'valeur_NO2': 13.3575,
        'valeur_O3': 17.6869,
        'valeur_PM10': 7.7189,
        'valeur_PM25': 4.9238,
        'overall': 8.7438
    }
    
    # Results from enhanced model (22 features)
    enhanced_results = {
        'valeur_CO': 0.0691,
        'valeur_NO2': 13.1428,
        'valeur_O3': 18.0608,
        'valeur_PM10': 8.1108,
        'valeur_PM25': 4.9238,
        'overall': 8.8615
    }
    
    print("BASIC MODEL (5 features):")
    print("  - hour, day_of_week, month, day_of_year, is_weekend")
    print("  - Fine-tuned with hyperparameter optimization")
    print("  - Temporal validation split")
    print()
    
    print("ENHANCED MODEL (22 features):")
    print("  - All basic features PLUS:")
    print("  - Cyclical encoding: hour_sin/cos, day_sin/cos, month_sin/cos")
    print("  - Interaction features: hour×day, hour×month, day×month, weekend×hour, weekend×month")
    print("  - Advanced features: time_since_midnight, is_monday, is_friday, end_of_month, etc.")
    print("  - Fine-tuned with hyperparameter optimization")
    print("  - Temporal validation split")
    print()
    
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Pollutant':<12} {'Basic MAE':<12} {'Enhanced MAE':<14} {'Improvement':<12} {'Status'}")
    print("-" * 80)
    
    improvements = []
    for pollutant in ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25']:
        basic_mae = basic_results[pollutant]
        enhanced_mae = enhanced_results[pollutant]
        improvement = ((basic_mae - enhanced_mae) / basic_mae) * 100
        
        if improvement > 0:
            status = "✅ BETTER"
        elif improvement > -2:
            status = "➖ SIMILAR"
        else:
            status = "❌ WORSE"
        
        improvements.append(improvement)
        print(f"{pollutant:<12} {basic_mae:<12.4f} {enhanced_mae:<14.4f} {improvement:<12.1f}% {status}")
    
    print("-" * 80)
    basic_overall = basic_results['overall']
    enhanced_overall = enhanced_results['overall']
    overall_improvement = ((basic_overall - enhanced_overall) / basic_overall) * 100
    
    if overall_improvement > 0:
        overall_status = "✅ BETTER"
    elif overall_improvement > -2:
        overall_status = "➖ SIMILAR"
    else:
        overall_status = "❌ WORSE"
    
    print(f"{'OVERALL':<12} {basic_overall:<12.4f} {enhanced_overall:<14.4f} {overall_improvement:<12.1f}% {overall_status}")
    print("=" * 80)
    
    # Analysis
    print("\nANALYSIS:")
    print("=" * 50)
    
    better_count = sum(1 for imp in improvements if imp > 0)
    worse_count = sum(1 for imp in improvements if imp < -2)
    similar_count = len(improvements) - better_count - worse_count
    
    print(f"Pollutants improved: {better_count}/5")
    print(f"Pollutants similar: {similar_count}/5") 
    print(f"Pollutants worse: {worse_count}/5")
    print()
    
    if overall_improvement > 1:
        print("🎉 FEATURE ENGINEERING SUCCESSFUL!")
        print("   Enhanced features provide meaningful improvement")
    elif overall_improvement > -1:
        print("➖ FEATURE ENGINEERING NEUTRAL")
        print("   Enhanced features provide minimal improvement")
    else:
        print("❌ FEATURE ENGINEERING UNSUCCESSFUL")
        print("   Enhanced features may be causing overfitting")
    
    print()
    print("KEY INSIGHTS:")
    print("-" * 30)
    
    # Find best and worst improvements
    best_improvement = max(improvements)
    worst_improvement = min(improvements)
    best_pollutant = ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25'][improvements.index(best_improvement)]
    worst_pollutant = ['valeur_CO', 'valeur_NO2', 'valeur_O3', 'valeur_PM10', 'valeur_PM25'][improvements.index(worst_improvement)]
    
    print(f"• Best improvement: {best_pollutant} ({best_improvement:.1f}%)")
    print(f"• Worst change: {worst_pollutant} ({worst_improvement:.1f}%)")
    print(f"• Overall change: {overall_improvement:.1f}%")
    
    if better_count >= 3:
        print("• Most pollutants benefited from feature engineering")
    elif worse_count >= 3:
        print("• Most pollutants were hurt by feature engineering (possible overfitting)")
    else:
        print("• Mixed results - some features help, others don't")
    
    print()
    print("RECOMMENDATIONS:")
    print("-" * 30)
    
    if overall_improvement > 1:
        print("✅ Use enhanced features for final submission")
        print("✅ Feature engineering was successful")
        print("✅ Consider this as your best model")
    elif overall_improvement > -1:
        print("➖ Enhanced features provide minimal benefit")
        print("➖ Consider using basic model for simplicity")
        print("➖ Feature engineering may not be worth the complexity")
    else:
        print("❌ Enhanced features are causing overfitting")
        print("❌ Stick with basic model")
        print("❌ Consider removing some features or using regularization")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if overall_improvement > 1:
        print("🎯 RECOMMENDATION: Use Enhanced Model")
        print(f"   Expected MAE on test set: {enhanced_overall:.4f}")
        print("   Enhanced features provide meaningful improvement")
    else:
        print("🎯 RECOMMENDATION: Use Basic Model")
        print(f"   Expected MAE on test set: {basic_overall:.4f}")
        print("   Basic model is more robust and less prone to overfitting")
    
    return {
        'basic_results': basic_results,
        'enhanced_results': enhanced_results,
        'overall_improvement': overall_improvement
    }

if __name__ == "__main__":
    results = compare_results()
