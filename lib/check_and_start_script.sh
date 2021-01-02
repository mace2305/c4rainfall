#!/bin/bash
pid_file='/home/yixiang/Documents/work_dir/c4rainfall/test/script.pid'
if [ ! -s "$pid_file" ] || ! kill -0 $(cat $pid_file) > /dev/null 2>&1; then
  echo $$ > "$pid_file"
  conda activate SOM
  exec /home/yixiang/Documents/work_dir/c4rainfall/lib/main.py
fi
 
