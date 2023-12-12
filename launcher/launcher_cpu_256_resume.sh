END=31
process_id=0
for ((i=0;i<=END;i++)); do
    gcloud alpha compute tpus tpu-vm ssh sfr-cxing-tpu-256-v4 --zone=us-central2-b --project salesforce-research-internal --worker=$i --command "bash /root/lemur-tpu/launcher/train_sf_256_70b_resume.sh ${i}" &
    process_id=$((process_id+1))
done