import argparse
from src.pipeline import ScenePipeline
from src.utils import setup_nltk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    setup_nltk()
    
    pipeline = ScenePipeline(base_path=args.data_dir)
    
    # Execute Pipeline
    pipeline.run_phase1_detection()
    pipeline.run_phase2_patching()
    pipeline.run_phase3_captioning()
    pipeline.run_phase4_reasoning()
    
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
