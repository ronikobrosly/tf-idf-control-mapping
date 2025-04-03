
## How to run the app

python run_control_mapping.py \
  --controls data/controls.csv \
  --name "comprehend" \
  --pdf_path "data/comprehend-dg.pdf" \
  --security_start_page 421 \
  --security_end_page 474 \
  --analyst_note "A resouce-based policy is attached to a custom model to authorize an entity from another account to replicate the model in their own account by importing it. An attacker can put a model policy and import the model into their own account." \
  --output output/results.json