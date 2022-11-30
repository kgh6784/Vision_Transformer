## ViT Architecture

<p align="center">
   <img src="Vision transformer.png" alt="vit"> 
</p>


## 스터디 with 준형 님(22/11/30)

Q1) 왜 batch norm이 아닌 layer norm을 사용할까?
- batch norm을 사용하는 이유 : In CNN for images, normalization within channel is helpful because weights are shared across channels.
=> 커널의 채널과 input feature map의 채널이 각각 곱해지면서 커널이 sliding되기 때문에 공유된다.
- layer norm을 사용하는 것은 ViT는 CNN과 달리 채널 간 weight를 공유하지 않고, 각 이미지별 attention이 곱해지기 때문인 것 같다. 
- CNN에 layer norm이나 instance norm을 하면 어떤가 했는데 이미 ConvNext에서 실험을 해보았다고 한다. 
Q2) FeedForward에서 왜 activation 함수를 한 번만 Linear 뒤에 쓰냐
<p align="center">
   <img src="Vision transformer.png" alt="vit"> 
</p>
Q3) 왜 MLP를 사용해야 하는가
- <Pay Attention to MLPs>를 보면, attention에서는 각 단어의 상관관계를 파악했으니(나는 그와의 관계를 '정리'했다, 집을 '정리'했다에서 정리는 각각 다른 의미), MLP에서는 고정된 파라미터를 바탕으로 전체적인 관계를 설명할 수 있는 모델을 만들기 위함이라고 되어 있다. 

Q4) q, k, v를 만들 때 linear로 해야 하는가?
   
Q5) early convolutions help transformers see better
   
Q6) learnable parameter - position embedding? 왜 randn?
   
