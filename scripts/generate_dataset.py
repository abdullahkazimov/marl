"""
Standalone script to generate synthetic datasets for MARLO training.
This can be run independently of the main CLI.
"""

import sys
import os

# Add src to path so we can import marlo modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from marlo.data.dataset_builder import DatasetBuilder
from marlo.utils.seed import set_seed
from marlo.utils.logger import get_logger


def main():
    """Generate a basic dataset for getting started."""
    logger = get_logger("dataset_generator")
    logger.info("Generating basic synthetic dataset")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create dataset builder
    builder = DatasetBuilder()
    
    # Generate synthetic dataset
    synthetic_path = "datasets/synthetic/basic_dataset.npz"
    metadata = builder.generate_synthetic_dataset(
        episodes=50,
        policy='random',
        output_path=synthetic_path,
        seed=42
    )
    
    print(f"✅ Synthetic dataset generated: {synthetic_path}")
    print(f"   Episodes: {metadata['dataset_info']['episodes']}")
    print(f"   Policy: {metadata['dataset_info']['policy']}")
    print(f"   Total transitions: {metadata['statistics']['total_transitions']}")
    
    # Generate semi-synthetic dataset
    semi_synthetic_path = "datasets/semi_synthetic/rush_hour_dataset.npz"
    metadata = builder.generate_semi_synthetic_dataset(
        episodes=30,
        traffic_patterns=['rush_hour', 'off_peak'],
        output_path=semi_synthetic_path,
        seed=42
    )
    
    print(f"✅ Semi-synthetic dataset generated: {semi_synthetic_path}")
    print(f"   Episodes: {metadata['dataset_info']['episodes']}")
    print(f"   Traffic patterns: {metadata.get('traffic_patterns', [])}")
    print(f"   Total transitions: {metadata['statistics']['total_transitions']}")
    
    logger.info("Dataset generation completed successfully")


if __name__ == "__main__":
    main()
