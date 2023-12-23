# datasets

this workload uses the huggingface datasets  
https://huggingface.co/docs/datasets/installation

install them with
```pip install datasets```

you can check the installation with (takes 1-2 mins)  
```python -c "from datasets import load_dataset; print(load_dataset('squad', split='train')[0])"```

you should see something like:  
```{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']}, 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.', 'id': '5733be284776f41900661182', 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?', 'title': 'University_of_Notre_Dame'}```

# transformer

this workload uses the huggingface transformers  
https://huggingface.co/docs/transformers/installation

install with
```pip install transformers```

test with (takes 1-2 mins)  
```python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"```

you should see something like:  
```[{'label': 'POSITIVE', 'score': 0.9998704195022583}]```

# tiktoken

https://github.com/openai/tiktoken^

- faster than hugging tokenizer (relevant, since its a long process ~10h? or smth? not finished yet)
- speed up of factor 2 - 3
- i have issues with parallelizing it, maybe its better to use hugging and give more workers

```pip install tiktoken```