mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=3 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=florida_gulf             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=all_nodes -P batch_size=2048 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.007503556066789443 -P number_of_dense_layers=2 -P number_of_rnn_layers=5 -P rnn_layer_size=91 -P dense_layer_size=263 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=3
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=3 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=florida_gulf             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=1024 -P dense_layer_size=65 -P direction=forward_only -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.0007636935263538555 -P number_of_dense_layers=0 -P number_of_rnn_layers=4 -P rnn_layer_size=141 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=1
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=3 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=florida_gulf             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=128 -P dense_layer_size=260 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=gru -P learning_rate=4.3377454427327665e-05 -P number_of_dense_layers=0 -P number_of_rnn_layers=5 -P rnn_layer_size=280 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=2

mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=2 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=florida_gulf             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=all_nodes -P batch_size=2048 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.007503556066789443 -P number_of_dense_layers=2 -P number_of_rnn_layers=5 -P rnn_layer_size=91 -P dense_layer_size=263 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=3
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=2 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=florida_gulf             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=1024 -P dense_layer_size=65 -P direction=forward_only -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.0007636935263538555 -P number_of_dense_layers=0 -P number_of_rnn_layers=4 -P rnn_layer_size=141 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=1
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=2 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=florida_gulf             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=128 -P dense_layer_size=260 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=gru -P learning_rate=4.3377454427327665e-05 -P number_of_dense_layers=0 -P number_of_rnn_layers=5 -P rnn_layer_size=280 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=2

mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=1 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=florida_gulf             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=all_nodes -P batch_size=2048 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.007503556066789443 -P number_of_dense_layers=2 -P number_of_rnn_layers=5 -P rnn_layer_size=91 -P dense_layer_size=263 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=3
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=1 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=florida_gulf             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=1024 -P dense_layer_size=65 -P direction=forward_only -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.0007636935263538555 -P number_of_dense_layers=0 -P number_of_rnn_layers=4 -P rnn_layer_size=141 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=1
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=1 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=florida_gulf             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=128 -P dense_layer_size=260 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=gru -P learning_rate=4.3377454427327665e-05 -P number_of_dense_layers=0 -P number_of_rnn_layers=5 -P rnn_layer_size=280 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=2


mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=3 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=new_york             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=all_nodes -P batch_size=2048 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.007503556066789443 -P number_of_dense_layers=2 -P number_of_rnn_layers=5 -P rnn_layer_size=91 -P dense_layer_size=263 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=3
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=3 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=new_york             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=1024 -P dense_layer_size=65 -P direction=forward_only -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.0007636935263538555 -P number_of_dense_layers=0 -P number_of_rnn_layers=4 -P rnn_layer_size=141 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=1
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=3 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=new_york             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=128 -P dense_layer_size=260 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=gru -P learning_rate=4.3377454427327665e-05 -P number_of_dense_layers=0 -P number_of_rnn_layers=5 -P rnn_layer_size=280 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=2

mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=2 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=new_york             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=all_nodes -P batch_size=2048 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.007503556066789443 -P number_of_dense_layers=2 -P number_of_rnn_layers=5 -P rnn_layer_size=91 -P dense_layer_size=263 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=3
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=2 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=new_york             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=1024 -P dense_layer_size=65 -P direction=forward_only -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.0007636935263538555 -P number_of_dense_layers=0 -P number_of_rnn_layers=4 -P rnn_layer_size=141 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=1
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=2 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=new_york             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=128 -P dense_layer_size=260 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=gru -P learning_rate=4.3377454427327665e-05 -P number_of_dense_layers=0 -P number_of_rnn_layers=5 -P rnn_layer_size=280 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=2

mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=1 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=new_york             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=all_nodes -P batch_size=2048 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.007503556066789443 -P number_of_dense_layers=2 -P number_of_rnn_layers=5 -P rnn_layer_size=91 -P dense_layer_size=263 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=3
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=1 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=new_york             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=1024 -P dense_layer_size=65 -P direction=forward_only -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.0007636935263538555 -P number_of_dense_layers=0 -P number_of_rnn_layers=4 -P rnn_layer_size=141 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=1
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=1 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=new_york             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=128 -P dense_layer_size=260 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=gru -P learning_rate=4.3377454427327665e-05 -P number_of_dense_layers=0 -P number_of_rnn_layers=5 -P rnn_layer_size=280 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=2



mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=3 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=california_coast             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=all_nodes -P batch_size=2048 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.007503556066789443 -P number_of_dense_layers=2 -P number_of_rnn_layers=5 -P rnn_layer_size=91 -P dense_layer_size=263 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=3
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=3 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=california_coast             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=1024 -P dense_layer_size=65 -P direction=forward_only -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.0007636935263538555 -P number_of_dense_layers=0 -P number_of_rnn_layers=4 -P rnn_layer_size=141 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=1
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=3 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=california_coast             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=128 -P dense_layer_size=260 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=gru -P learning_rate=4.3377454427327665e-05 -P number_of_dense_layers=0 -P number_of_rnn_layers=5 -P rnn_layer_size=280 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=2

mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=2 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=california_coast             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=all_nodes -P batch_size=2048 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.007503556066789443 -P number_of_dense_layers=2 -P number_of_rnn_layers=5 -P rnn_layer_size=91 -P dense_layer_size=263 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=3
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=2 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=california_coast             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=1024 -P dense_layer_size=65 -P direction=forward_only -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.0007636935263538555 -P number_of_dense_layers=0 -P number_of_rnn_layers=4 -P rnn_layer_size=141 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=1
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=2 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=california_coast             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=128 -P dense_layer_size=260 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=gru -P learning_rate=4.3377454427327665e-05 -P number_of_dense_layers=0 -P number_of_rnn_layers=5 -P rnn_layer_size=280 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=2

mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=1 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=california_coast             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=all_nodes -P batch_size=2048 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.007503556066789443 -P number_of_dense_layers=2 -P number_of_rnn_layers=5 -P rnn_layer_size=91 -P dense_layer_size=263 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=3
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=1 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=california_coast             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=1024 -P dense_layer_size=65 -P direction=forward_only -P distance_traveled=ignore -P layer_type=lstm -P learning_rate=0.0007636935263538555 -P number_of_dense_layers=0 -P number_of_rnn_layers=4 -P rnn_layer_size=141 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=1
mlflow run . -e test_time_gaps -P time_gap=30 -P length_of_history=3 -P hours_out=1 -P loss=haversine -P time_of_day=hour_day -P weather=currents -P dataset_name=california_coast             --experiment-name 'Final' -P model_type=long_term_fusion -P sog_cog=raw -P rnn_to_dense_connection=final_node -P batch_size=128 -P dense_layer_size=260 -P direction=bidirectional -P distance_traveled=ignore -P layer_type=gru -P learning_rate=4.3377454427327665e-05 -P number_of_dense_layers=0 -P number_of_rnn_layers=5 -P rnn_layer_size=280 -P extended_recurrent_idxs=vt_dst_and_time -P number_of_fusion_weather_layers=2

