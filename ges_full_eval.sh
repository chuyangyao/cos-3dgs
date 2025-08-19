# python train_ges.py -s ../autodl-tmp/nerf_360/bicycle -i images_4 -m ../autodl-tmp/eval/ges_eval/bicycle --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/nerf_360/flowers -i images_4 -m ../autodl-tmp/eval/ges_eval/flowers --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/nerf_360/garden -i images_4 -m ../autodl-tmp/eval/ges_eval/garden --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/nerf_360/stump -i images_4 -m ../autodl-tmp/eval/ges_eval/stump --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/nerf_360/treehill -i images_4 -m ../autodl-tmp/eval/ges_eval/treehill --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/nerf_360/room -i images_2 -m ../autodl-tmp/eval/ges_eval/room --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/nerf_360/counter -i images_2 -m ../autodl-tmp/eval/ges_eval/counter --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/nerf_360/kitchen -i images_2 -m ../autodl-tmp/eval/ges_eval/kitchen --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/nerf_360/bonsai -i images_2 -m ../autodl-tmp/eval/ges_eval/bonsai --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/tandt_db/tandt/truck -m ../autodl-tmp/eval/ges_eval/truck --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/tandt_db/tandt/train -m ../autodl-tmp/eval/ges_eval/train --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/tandt_db/db/drjohnson -m ../autodl-tmp/eval/ges_eval/drjohnson --quiet --eval --test_iterations -1 
# python train_ges.py -s ../autodl-tmp/tandt_db/db/playroom -m ../autodl-tmp/eval/ges_eval/playroom --quiet --eval --test_iterations -1 

# python render.py --iteration 7000 -s ../autodl-tmp/nerf_360/bicycle -m ../autodl-tmp/eval/ges_eval/bicycle --quiet --eval --skip_train --render_dir ges
# python render.py --iteration 30000 -s ../autodl-tmp/nerf_360/bicycle -m ../autodl-tmp/eval/ges_eval/bicycle --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/nerf_360/flowers -m ../autodl-tmp/eval/ges_eval/flowers --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/nerf_360/flowers -m ../autodl-tmp/eval/ges_eval/flowers --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/nerf_360/garden -m ../autodl-tmp/eval/ges_eval/garden --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/nerf_360/garden -m ../autodl-tmp/eval/ges_eval/garden --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/nerf_360/stump -m ../autodl-tmp/eval/ges_eval/stump --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/nerf_360/stump -m ../autodl-tmp/eval/ges_eval/stump --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/nerf_360/treehill -m ../autodl-tmp/eval/ges_eval/treehill --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/nerf_360/treehill -m ../autodl-tmp/eval/ges_eval/treehill --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/nerf_360/room -m ../autodl-tmp/eval/ges_eval/room --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/nerf_360/room -m ../autodl-tmp/eval/ges_eval/room --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/nerf_360/counter -m ../autodl-tmp/eval/ges_eval/counter --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/nerf_360/counter -m ../autodl-tmp/eval/ges_eval/counter --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/nerf_360/kitchen -m ../autodl-tmp/eval/ges_eval/kitchen --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/nerf_360/kitchen -m ../autodl-tmp/eval/ges_eval/kitchen --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/nerf_360/bonsai -m ../autodl-tmp/eval/ges_eval/bonsai --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/nerf_360/bonsai -m ../autodl-tmp/eval/ges_eval/bonsai --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/tandt_db/tandt/truck -m ../autodl-tmp/eval/ges_eval/truck --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/tandt_db/tandt/truck -m ../autodl-tmp/eval/ges_eval/truck --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/tandt_db/tandt/train -m ../autodl-tmp/eval/ges_eval/train --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/tandt_db/tandt/train -m ../autodl-tmp/eval/ges_eval/train --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/tandt_db/db/drjohnson -m ../autodl-tmp/eval/ges_eval/drjohnson --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/tandt_db/db/drjohnson -m ../autodl-tmp/eval/ges_eval/drjohnson --quiet --eval --skip_train --render_dir ges
#python render.py --iteration 7000 -s ../autodl-tmp/tandt_db/db/playroom -m ../autodl-tmp/eval/ges_eval/playroom --quiet --eval --skip_train --render_dir ges
python render.py --iteration 40000 -s ../autodl-tmp/tandt_db/db/playroom -m ../autodl-tmp/eval/ges_eval/playroom --quiet --eval --skip_train --render_dir ges

python metrics_debug.py -m "../autodl-tmp/eval/ges_eval/bicycle" "../autodl-tmp/eval/ges_eval/flowers" "../autodl-tmp/eval/ges_eval/garden" "../autodl-tmp/eval/ges_eval/stump" "../autodl-tmp/eval/ges_eval/treehill" "../autodl-tmp/eval/ges_eval/room" "../autodl-tmp/eval/ges_eval/counter" "../autodl-tmp/eval/ges_eval/kitchen" "../autodl-tmp/eval/ges_eval/bonsai" "../autodl-tmp/eval/ges_eval/truck" "../autodl-tmp/eval/ges_eval/train" "../autodl-tmp/eval/ges_eval/drjohnson" "../autodl-tmp/eval/ges_eval/playroom"
