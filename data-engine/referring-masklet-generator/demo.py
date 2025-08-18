import os
import joblib
import subprocess
from pathlib import Path


strefer_dir = Path(__file__).resolve().parent.parent.parent


def referring_masklet_generator(expressions):
    joblib.dump(expressions, f'{strefer_dir}/data-engine/referring-masklet-generator/tmp.pkl')

    subprocess.run(
        f"bash {strefer_dir}/data-engine/referring-masklet-generator/Grounded-SAM-2/1.get_tracklet.sh", 
        shell=True)
    subprocess.run(
        f"bash {strefer_dir}/data-engine/referring-masklet-generator/RexSeek/2.get_referring_matching.sh", 
        shell=True)
    subprocess.run(
        f"bash {strefer_dir}/data-engine/referring-masklet-generator/Grounded-SAM-2/3.vis_tracklet.sh", 
        shell=True)

    check_success_status()
    return


def check_success_status():
    SUCCESS = False
    TMP_DIR = f'{strefer_dir}/data-engine/referring-masklet-generator/Grounded-SAM-2/output/tmp'
    if os.path.exists(
        os.path.join(
            TMP_DIR, 
            "referring_track/referring_expression_masklets_results.png")):
        SUCCESS = True
        referring_tracklets = joblib.load(
            os.path.join(TMP_DIR, 'referring_tracklets.pkl'))
        
    
    print('%'*100)
    if SUCCESS:
        print("SUCCESS")
    else:
        print("FAILED <--- NOTE")
    return


if __name__ == "__main__":
    
    expressions = {
        'referrings': ['child in a white T-shirt', 
                       'child in a pink top', 
                       'dog with a red leash', 
                       'woman with ponytail'],
        'generalized_nouns': ['person', 
                              'person', 
                              'dog', 
                              'person'],
        'video_path': f'{strefer_dir}/assets/test-referring_masklet_generator.mp4'
    }
    print(expressions)
    referring_masklet_generator(expressions)
    print("\n\nFINISHED!\n")
    
    