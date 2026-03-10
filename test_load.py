from bertopic import BERTopic
import warnings
warnings.filterwarnings('ignore')

print("Loading model...")
topic_model = BERTopic.load('results/topic_models/bertopic_abpa_2000-2025_V_pp3')
print('模型加载成功!')
print(f'主题数量: {len(topic_model.get_topic_info()) - 1}')
