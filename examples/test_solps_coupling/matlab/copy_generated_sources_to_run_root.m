function copy_generated_sources_to_run_root(src_dir, run_dir)
if ~isfolder(run_dir)
    error('Run directory not found: %s', run_dir);
end
delete(fullfile(run_dir, 'source2d.*'));
delete(fullfile(run_dir, 'b2.sources.profile*'));
copyfile(fullfile(src_dir, 'source2d.*'), run_dir);
copyfile(fullfile(src_dir, 'b2.sources.profile*'), run_dir);
if exist(fullfile(src_dir, 'sources_time_windows.txt'), 'file')
    copyfile(fullfile(src_dir, 'sources_time_windows.txt'), run_dir);
end
fprintf('Deployed source2d.* and b2.sources.profile* from %s -> %s\n', src_dir, run_dir);
end
