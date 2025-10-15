"""
Fix Kaggle Submission Format
===========================

This script fixes the id format in the submission file to match Kaggle's expected format.
"""

import pandas as pd

def fix_kaggle_submission():
    """Fix the id format in the submission file"""
    
    print("=" * 60)
    print("FIXING KAGGLE SUBMISSION FORMAT")
    print("=" * 60)
    
    # Load the current submission
    print("1. Loading current submission...")
    submission = pd.read_csv('kaggle_submission_final.csv')
    
    print(f"   Current format: {submission['id'].iloc[0]}")
    print(f"   Total rows: {len(submission)}")
    
    # Convert datetime to the expected format (YYYY-MM-DD HH)
    print("\n2. Converting id format...")
    submission['id'] = pd.to_datetime(submission['id']).dt.strftime('%Y-%m-%d %H')
    
    print(f"   New format: {submission['id'].iloc[0]}")
    
    # Save the fixed submission
    print("\n3. Saving fixed submission...")
    submission.to_csv('kaggle_submission_fixed.csv', index=False)
    
    print(f"   Saved as: kaggle_submission_fixed.csv")
    print(f"   Rows: {len(submission)}")
    print(f"   Columns: {list(submission.columns)}")
    
    # Show sample of fixed submission
    print("\n4. Sample of fixed submission:")
    print(submission.head().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("KAGGLE SUBMISSION FIXED SUCCESSFULLY!")
    print("=" * 60)
    print("✅ ID format converted to 'YYYY-MM-DD HH'")
    print("✅ Ready for Kaggle submission")
    print("✅ File: kaggle_submission_fixed.csv")
    print("=" * 60)
    
    return submission

if __name__ == "__main__":
    fix_kaggle_submission()
