export MAGNUM_LOG=quiet GLOG_minloglevel=2
python -u main_vis_multi.py -n 4 -v 1 --fmm_planner &> batch.log
# python -u main_vis.py -v 1 --fmm_planner &> batch.log

# tmux new-session 'bash -c "conda activate webvnc && bash /startup.sh"' -s webvnc -d
# tmux new-session -d -s vis_multi "python main_vis_multi.py -n 3 -v 1 --fmm_planner" >
# deattach: Ctrl+B. After that, press D
# tmux attach -t 0