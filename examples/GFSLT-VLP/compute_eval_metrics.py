try:
    from nlgeval import compute_metrics
except:
    print('Please install nlgeval package.')

output_dir = "./out/SignIR_noSignCL_eval"
metrics_dict = compute_metrics(hypothesis=output_dir+'/tmp_pres.txt',
                    references=[output_dir+'/tmp_refs.txt'],no_skipthoughts=True,no_glove=True)
print('*'*80)
print(metrics_dict)