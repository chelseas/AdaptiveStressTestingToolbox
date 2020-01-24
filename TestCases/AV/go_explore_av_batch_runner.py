from TestCases.AV.go_explore_av_runner import runner

if __name__ == '__main__':
    use_ram = False
    db_filename = '/home/mkoren/scratch/data/test3'
    max_db_size = 150
    overwrite_db = True
    n_parallel = 8
    snapshot_mode = 'last'
    snapshot_gap = 1
    log_dir = '/home/mkoren/scratch/data/test3'
    max_path_length = 50
    discount = 0.99
    n_itr = 100
    n_itr_robust = 1
    max_kl_step = 1.0
    whole_paths = False
    batch_size = 5000
    batch_size_robust = 5000

    run_1 = runner(exp_name='av',
                            use_ram=use_ram,
                            db_filename=db_filename,
                            max_db_size=max_db_size,
                            overwrite_db=overwrite_db,
                            n_parallel=n_parallel,
                            snapshot_mode=snapshot_mode,
                            snapshot_gap=snapshot_gap,
                            log_dir=log_dir,
                            max_path_length=max_path_length,
                            discount=discount,
                            n_itr=n_itr,
                            n_itr_robust=n_itr_robust,
                            max_kl_step=max_kl_step,
                            whole_paths=whole_paths,
                            batch_size=batch_size,
                            batch_size_robust=batch_size_robust)