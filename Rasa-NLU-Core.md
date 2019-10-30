<h1><center>  RASA NLU </center> </h1>

#### 基本操作

​	聊天机器人和AI助手的语言理解。Rasa NLU是一种开源自然语言处理工具，用于聊天机器人中的意图识别，响应检索和实体抽取。例如

```
"I am looking for a Mexican restaurant in the center of town"
```

意图识别和实体抽取后结构化数据如下，

```
{
  "intent": "search_restaurant",
  "entities": {
    "cuisine" : "Mexican",
    "location" : "center"
  }
}
```

Rasa NLU曾经是一个单独的库，但现在是Rasa框架的一部分。

我们可以仅仅使用RASA训练NLU模型，只要运行如下命令：

```
rasa train nlu
```

这将在`data/`目录中查找NLU训练数据文件，并将训练后的模型保存在`models/`目录中。模型的名字将以`nlu-`开头。

训练好模型之后，让我们来测试一下NLU模型，使用以下命令

```
rasa shell nlu
```

这将启动rasa shell，并要求输入消息进行测试。另外，可以省略`nlu`参数并直接传入NLU模型：

```
rasa shell -m models/nlu-20191007-094608.tar.gz
```

要使用NLU模型启动服务器，在运行时传递模型名字：

```
rasa run --enable-api -m models/nlu-20191007-094608.tar.gz
```

然后，我们可以访问`localhost:5005/model/parse` URL请求。为此，运行下面的例子：

```
curl localhost:5005/model/parse -d '{"text":"hello"}'
```



#### 训练数据格式

**数据格式**: 支持Markdown或JSON，单个文件或包含多个文件的目录的形式提供训练数据。但是，我们知道Markdown通常更易于使用。

**Markdown Format**：Markdown是我们最容易阅读和书写的Rasa NLU格式。有对Markdown语法不熟悉的可以查阅相关资料了解一下，整体是非常简洁的。实例按意图分组，而实体则标注为Markdown链接，例如。`[entity](entity name)`

```
## intent:check_balance
- what is my balance <!-- no entity -->
- how much do I have on my [savings](source_account) <!-- entity "source_account" has value "savings" -->
- how much do I have on my [savings account](source_account:savings) <!-- synonyms, method 1-->
- Could I pay in [yen](currency)?  <!-- entity matched by lookup table -->

## intent:greet
- hey
- hello

## synonym:savings   <!-- synonyms, method 2 -->
- pink pig

## regex:zipcode
- [0-9]{5}

## lookup:currencies   <!-- lookup table list -->
- Yen
- USD
- Euro

## lookup:additional_currencies  <!-- no list to specify lookup table file -->
path/to/currencies.txt
```

Rasa NLU的训练数据分为以下几个部分：

- 通用示例
- 同义词
- 正则表达式
- 查找表

虽然通用示例是唯一必需的部分，但包括其他示例将帮助NLU模型更好的学习领域，并有助于对其预测更有帮助。

​	同义词会将提取的实体映射到相同的名称，例如，将“我的储蓄帐户”映射为简单的“储蓄”。但是，这仅*在*提取实体*之后*才发生，因此需要提供带有存在同义词的示例，以便Rasa可以学习并将其提取。

​	查找表可以直接指定为列表，也可以指定为包含换行符分隔的单词或短语的txt文件。加载训练数据后，这些文件用于生成不区分大小写的正则表达式模式，该模式会添加到正则表达式功能中。例如，在这种情况下，将提供名称列表，以便更轻松地选择该实体。

**JSON格式**：JSON格式由被称为顶层对象的`rasa_nlu_data`，与键`common_examples` `entity_synonyms`和`regex_features`。最重要的是`common_examples`。

```
{
    "rasa_nlu_data": {
        "common_examples": [],
        "regex_features" : [],
        "lookup_tables"  : [],
        "entity_synonyms": []
    }
}
```

`common_examples`是用来训练模型。将所有训练示例放在`common_examples`数组中。正则表达式功能是帮助分类器检测实体或意图并提高性能的工具。

##### 提高意图分类和实体识别

**通用示例**: 由三个组成部分：`text`，`intent`和`entities`。前两个是字符串，最后一个是数组。

> - 该*文本*是用户消息[必须]
> - *意图*是，应与文字相关的意图[可选]
> - 该*实体*是需要被识别的文本的特定部分[可选]

实体用`start`和`end`值指定，指定了实体开始和结束的位置，例如在下面的示例中，text="show me chinese restaurants"，text[8:15] == 'chinese'。实体可以跨越多个单词，实际上，该字段不必与示例中的子字符串完全对应。这样，可以将同义词或拼写错误映射到同一个。

```
## intent:restaurant_search
- show me [chinese](cuisine) restaurants
```

**正则表达式**： 正则表达式可用于支持意图分类和实体提取。例如，如果实体具有确定性结构（例如邮政编码或电子邮件地址），则可以使用正则表达式来简化对该实体的抽取。对于邮政编码示例，它可能如下所示：

```
## regex:zipcode
- [0-9]{5}

## regex:greet
- hey[^\\s]*
```

名称没有定义实体，也没有定义意图，它只是人类可读的描述，可以记住该正则表达式的用途，并且是相应模式特征的标题。如上例所示，还可以使用正则表达式功能来改善意图分类性能。

尝试以使其与尽可能少的单词匹配的方式创建正则表达式。例如，使用`hey[^\s]*` 而不是`hey.*`，因为后一个可能匹配整个消息，而第一个可能只匹配一个单词。

目前只有`CRFEntityExtractor`组件支持用于实体提取的正则表达式功能！因此，其他实体提取器像`MitieEntityExtractor`或`SpacyEntityExtractor`不使用生成的特征，并且它们的存在不会提高这些提取器的实体识别度。当前，所有意图分类器都支持正则表达式功能。

**查找表**：训练数据中也可以指定外部文件形式的查找表或元素列表。外部提供的查找表必须采用换行符分隔。例如，`data/test/lookup_tables/plates.txt`可能包含：

```
tacos
beef
mapo tofu
burrito
lettuce wrap
```

可以加载为：

```
## lookup:plates
data/test/lookup_tables/plates.txt
```

或者，可以将查找元素直接包含在列表中

```
## lookup:plates
- beans
- rice
- tacos
- cheese
```

在训练数据中提供查找表时，内容将组合成一个大写，不区分大小写的正则表达式模式，该模式在训练示例中查找完全匹配的内容。这些正则表达式可匹配多个，因此lettuce wrap将匹配`get me a lettuce wrap ASAP`为`[0 0 0 1 1 0]`。这些正则表达式的处理方式与直接在训练数据中指定的常规正则表达式样式相同。

##### 数据标准化

**实体同义词**：如果将实体定义为具有相同的值，则它们将被视为同义词。这是一个例子：

```
## intent:search
- in the center of [NYC](city:New York City)
- in the centre of [New York City](city)
```

如你所见，在两个示例中均`city`具有值。通过将value属性定义为与实体的开始索引和结束索引之间的文本中找到的值不同，可以定义同义词。每当找到相同的文本时，该值将使用同义词代替消息中的实际文本。

要使用训练数据中定义的同义词，您需要确保管道包含`EntitySynonymMapper` 组件。

另外，您可以添加一个“ entity_synonyms”数组来为一个实体值定义多个同义词。如下：

```
## synonym:New York City
- NYC
- nyc
- the big apple
```
有时生成一堆实体示例会很有帮助，例如，如果您有餐厅名称数据库。社区构建了一些工具来帮助实现这一目标。可以使用[Chatito](https://rodrigopivi.github.io/Chatito/)为rasa 创建训练数据集。但是，创建综合示例通常会导致过拟合，如果您有大量实体值，则最好使用查找表。



#### 选择Pipeline

选择NLU Pipeline可以自定义模型并在数据集上进行微调。如果训练数据少于1000，并且语言支持spaCy模型，请使用`pretrained_embeddings_spacy Pipeline`：

```
language: "en"
pipeline: "pretrained_embeddings_spacy"
```

如果您有1000或更多带标签训练数据，请使用`supervised_embeddings Pipeline`：

```
language: "en"

pipeline: "supervised_embeddings"
```

最重要的两个管道是`supervised_embeddings`和`pretrained_embeddings_spacy`。它们之间的最大区别是`pretrained_embeddings_spacy`使用来自GloVe或fastText的预训练词向量。但是，`supervised_embeddings`不使用任何预先训练的词向量。

`pretrained_embeddings_spacy` pipeline的优势在于，如果您有一个训练数据，例如：“I want to buy apples”，并且要求Rasa预测“get pears”的意图，那么模型已经知道“apples”和“pears”这两个词非常相似。如果没有太多的训练数据，这将特别有用。

`supervised_embeddings` pipeline的优势在于，将针对领域自定义单词向量。例如，在通用英语中，“balance”一词与“symmetry”密切相关，但与“cash”一词有很大不同。在银行领域中，“balance”和“cash”密切相关，希望模型能够做到这一点。该pipeline不使用特定于语言的模型，因此它将与您可以标记化的任何语言一起使用。

也可以在管道中使用MITIE作为单词向量的来源，MITIE后端对于小型数据集表现良好，但是如果有数百个以上的数据，则训练可能会花费很长时间。不建议使用它，因为在将来的版本中可能会不再支持mitie支持。


##### 类不平衡
如果类别失衡很大，例如，如果有很多针对某些意图的训练数据而很少有针对其他意图的训练数据，则分类算法通常不会表现良好。为了缓解此问题，rasa的`supervised_embeddings`管道使用了`balanced`批处理策略。该算法确保在每个批次中或至少在尽可能多的后续批次中代表所有类别，仍然模仿某些类别比其他类别更频繁的事实。默认情况下使用平衡批处理。为了将其关闭并使用经典的批处理策略，请在您的配置文件中添加该策略 。`batch_strategy: sequence`

```
language: "en"

pipeline:
- name: "CountVectorsFeaturizer"
- name: "EmbeddingIntentClassifier"
  batch_strategy: sequence
```

##### 多意图
如果要将意图拆分为多个标签（例如，用于预测多个意图或为分层意图结构建模），则只能使用受监督的嵌入管道来执行此操作。为此，请在中使用这些标志：`Whitespace Tokenizer`

> - `intent_split_symbol`：设置分隔符字符串以拆分意图标签。默认`_`

```
language: "en"

pipeline:
- name: "WhitespaceTokenizer"
  intent_split_symbol: "_"
- name: "CountVectorsFeaturizer"
- name: "EmbeddingIntentClassifier"
```

##### 了解Rasa NLU管道
在Rasa NLU中，传入消息由一系列组件处理。这些组件在所谓的处理管道中一个接一个地执行。有用于实体提取，意图分类，响应选择，预处理等的组件，也支持自定义组件。

每个组件都处理输入并创建输出。输出可以由管道中该组件之后的任何组件使用。有一些组件仅生成流水线中其他组件使用的信息，还有其他一些组件会生成`Output`属性，这些属性将在处理完成后返回。例如，对于该句子，输出为：`"I am looking for Chinese food"`

```
{
    "text": "I am looking for Chinese food",
    "entities": [
        {"start": 8, "end": 15, "value": "chinese", "entity": "cuisine", "extractor": "CRFEntityExtractor", "confidence": 0.864}
    ],
    "intent": {"confidence": 0.6485910906220309, "name": "restaurant_search"},
    "intent_ranking": [
        {"confidence": 0.6485910906220309, "name": "restaurant_search"},
        {"confidence": 0.1416153159565678, "name": "affirm"}
    ]
}
```

这是预配置管道中不同组件的结果的组合`pretrained_embeddings_spacy`。例如，`entities`属性是由`CRFEntityExtractor`组件创建的。

##### 预先配置的管道
模板只是完整组件列表的快捷方式。例如，这两个配置是等效的：

```
language: "en"

pipeline: "pretrained_embeddings_spacy"
language: "en"

pipeline:
- name: "SpacyNLP"
- name: "SpacyTokenizer"
- name: "SpacyFeaturizer"
- name: "RegexFeaturizer"
- name: "CRFEntityExtractor"
- name: "EntitySynonymMapper"
- name: "SklearnIntentClassifier"
```

以下是所有带有定制信息的预配置管道模板的列表。

**supervised_embeddings**：要以首选语言训练Rasa模型，请`supervised_embeddings`在您`config.yml`或其他配置文件中将管道定义为 管道：

```
language: "en"

pipeline: "supervised_embeddings"
```

该`supervised_embeddings`管道的支持可以符号化的任何语言。默认情况下，它使用空格进行标记化。可以通过添加或更改组件来自定义此管道的设置。以下是构成`supervised_embeddings`管道的默认组件：

```
language: "en"

pipeline:
- name: "WhitespaceTokenizer"
- name: "RegexFeaturizer"
- name: "CRFEntityExtractor"
- name: "EntitySynonymMapper"
- name: "CountVectorsFeaturizer"
- name: "CountVectorsFeaturizer"
  analyzer: "char_wb"
  min_ngram: 1
  max_ngram: 4
- name: "EmbeddingIntentClassifier"
```

因此，例如，如果您选择的语言没有使用空格标记（单词之间没有空格），则可以用`WhitespaceTokenizer`自己的标记器替换。管道使用的两个实例`CountVectorsFeaturizer`。第一个将基于单词的文本特征化。第二个基于字符n-gram对文本进行特征化处理，保留单词边界。

**pretrained_embeddings_spacy**：要使用`pretrained_embeddings_spacy`模板：

```
language: "en"
pipeline: "pretrained_embeddings_spacy"

language: "en"
pipeline:
- name: "SpacyNLP"
- name: "SpacyTokenizer"
- name: "SpacyFeaturizer"
- name: "RegexFeaturizer"
- name: "CRFEntityExtractor"
- name: "EntitySynonymMapper"
- name: "SklearnIntentClassifier"
```

**自定义管道**：也可以不必使用模板，通过列出要使用的组件的名称来运行完全自定义的管道：

```
pipeline:
- name: "SpacyNLP"
- name: "CRFEntityExtractor"
- name: "EntitySynonymMapper"
```

这将创建仅执行实体识别但不进行意图分类的管道。因此，Rasa NLU不会预测任何意图。

#### 语言支持

可以使用Rasa以所需的任何语言构建助手！Rasa的 `supervised_embeddings` pipeline可以用于**任何语言的**训练数据。该pipeline使用提供的数据从头开始创建单词嵌入。此外，我们还支持预训练的单词嵌入，例如spaCy。

**支持任何语言训练模型**：Rasa的`supervised_embeddings` pipeline可用于以任何语言训练模型，因为它使用自己的训练数据来创建自定义单词嵌入。这意味着任何特定单词的向量表示形式将取决于其与训练数据中其他单词的关系。这种定制还意味着，该pipeline非常适合依赖特定于域的数据的用例，例如那些需要提取特定产品名称的用例。

要使用首选语言训练Rasa模型，在`config.yml`中设置`supervised_embeddings`。
定义`supervised_embeddings`处理pipeline并 使用该语言生成一些NLU训练数据后，使用下面命令训练。

```
rasa shell nlu
```

#### 实体提取
##### 介绍

以下是可用提取器及其用途的介绍：

| Component               | Requires          | Model                    | Notes                             |
| ----------------------- | ----------------- | ------------------------ | --------------------------------- |
| `CRFEntityExtractor`    | sklearn-crfsuite  | conditional random field | good for training custom entities |
| `SpacyEntityExtractor`  | spaCy             | averaged perceptron      | provides pre-trained entities     |
| `DucklingHTTPExtractor` | running duckling  | context-free grammar     | provides pre-trained entities     |
| `MitieEntityExtractor`  | MITIE             | structured SVM           | good for training custom entities |
| `EntitySynonymMapper`   | existing entities | N/A                      | maps known synonyms               |

如果pipeline包括上述一个或多个组件，则经过训练的模型的输出将包括提取的实体以及有关哪个组件提取了它们的一些元数据。该`processors`字段包含更改每个实体的组件的名称。
这是一个示例响应：

```
{
  "text": "show me chinese restaurants",
  "intent": "restaurant_search",
  "entities": [
    {
      "start": 8,
      "end": 15,
      "value": "chinese",
      "entity": "cuisine",
      "extractor": "CRFEntityExtractor",
      "confidence": 0.854,
      "processors": []
    }
  ]
}
```

某些提取器（如`duckling`）可能包含其他信息。例如：

```
{
  "additional_info":{
    "grain":"day",
    "type":"value",
    "value":"2018-06-21T00:00:00.000-07:00",
    "values":[
      {
        "grain":"day",
        "type":"value",
        "value":"2018-06-21T00:00:00.000-07:00"
      }
    ]
  },
  "confidence":1.0,
  "end":5,
  "entity":"time",
  "extractor":"DucklingHTTPExtractor",
  "start":0,
  "text":"today",
  "value":"2018-06-21T00:00:00.000-07:00"
}
```

##### 自定义实体
几乎每个聊天机器人和语音应用程序都会有一些自定义实体。餐饮助手应该`chinese`理解为美食，但是对于语言学习助手来说，意义却大不相同。`CRFEntityExtractor`给定一些训练数据，该组件可以使用任何语言学习自定义实体。

**提取地点，日期，人名，组织**：spaCy具有针对几种不同语言的出色的经过预先训练的命名实体识别器。请注意，某些spaCy模型高度区分大小写。

**日期，金额，期限，距离，序号**：duckling库做了一下转换，如“next Thursday at 8pm”表述为实际的datetime对象

```
"next Thursday at 8pm" => {"value":"2018-05-31T20:00:00.000+01:00"}
```
##### 正则表达式(regex)
可以使用正则表达式来帮助CRF模型学习识别实体。在训练数据中提供一个正则表达式列表，每个正则表达式都提供一个`CRFEntityExtractor`带有额外的二进制功能的正则表达式，该正则表达式说明是否找到了正则表达式（1）（0）。如果只想精确匹配正则表达式，则可以在收到Rasa NLU的响应后，在代码中执行此操作，作为后处理步骤。

#### 组件
这是Rasa NLU中每个内置组件的配置选项的参考。

|Components|type|
|:--:|:--:|
|Word Vector Sources|MitieNLP<br />SpacyNLP|
|Featurizers|MitieFeaturizer<br /> SpacyFeaturizer<br />NGramFeaturizer,<br />RegexFeaturizer<br />CountVectorsFeaturizer|
|Intent Classifiers|KeywordIntentClassifier<br />MitieIntentClassifier<br />SklearnIntentClassifier<br />EmbeddingIntentClassifier|
|Selectors|Response Selector|
|Tokenizers|WhitespaceTokenizer<br />JiebaTokenizer<br /> MitieTokenizer<br />SpacyTokenizer|
|Entity Extractors|MitieEntityExtractor<br />SpacyEntityExtractor<br /> EntitySynonymMapper<br />CRFEntityExtractor<br />DucklingHTTPExtractor|

##### Word Vector Sources
**MitieNLP**

| Short:         | MITIE initializer                                            |
| -------------- | ------------------------------------------------------------ |
| Outputs:       | nothing                                                      |
| Requires:      | nothing                                                      |
| Description:   | 初始化mitie结构。每个mitie组件都依赖于此，因此应将其放在使用任何mitie组件的每个pipeline的开头。 |
| Configuration: | MITIE库需要一个语言模型文件，该文件**必须**在配置中指定：    |

```
pipeline:
- name: "MitieNLP"
  # language model to load
  model: "data/total_word_feature_extractor.dat"
```

**SpacyNLP**

| Short:         | spacy language initializer                                   |
| -------------- | ------------------------------------------------------------ |
| Outputs:       | nothing                                                      |
| Requires:      | nothing                                                      |
| Description:   | 初始化spacy结构。每个spacy组件都依赖于此，因此应将其放在使用任何spacy组件的每个pipeline的开头。 |
| Configuration: | 语言模型，默认情况下将使用配置的语言。如果要使用的模型spacy具有名称是从语言标签（不同`"en"`，`"de"`等），可使用此配置变量指定的型号名称。该名称将传递给`spacy.load(name)`。 |

```
pipeline:
- name: "SpacyNLP"
  # language model to load
  model: "en_core_web_md"

  # when retrieving word vectors, this will decide if the casing
  # of the word is relevant. E.g. `hello` and `Hello` will
  # retrieve the same vector, if set to `false`. For some
  # applications and models it makes sense to differentiate
  # between these two words, therefore setting this to `true`.
  case_sensitive: false
```



##### Featurizers

**MitieFeaturizer**
| Short:         | MITIE intent featurizer                                      |
| -------------- | ------------------------------------------------------------ |
| Outputs:       | nothing, 用作需要意图特征的意图分类器的输入（例如`SklearnIntentClassifier`） |
| Requires:      | [MitieNLP](https://rasa.com/docs/rasa/nlu/components/#mitienlp) |
| Description:   | 使用MITIE featurizer创建用于意图分类的功能。注意不使用的`MitieIntentClassifier`组件。当前，仅`SklearnIntentClassifier`能够使用预先计算的功能。 |
| Configuration: | pipeline: <br />- name: "MitieFeaturizer"                    |



###### SpacyFeaturizer
| Short:       | spacy intent featurizer                                      |
| ------------ | ------------------------------------------------------------ |
| Outputs:     | nothing, 用作需要意图特征的意图分类器的输入（例如`SklearnIntentClassifier`） |
| Requires:    | [SpacyNLP](https://rasa.com/docs/rasa/nlu/components/#spacynlp) |
| Description: | 使用spacy featurizer创建用于意图分类的功能。                 |

###### NGramFeaturizer
| Short:         | 将字符特征附加到特征向量                                     |
| -------------- | ------------------------------------------------------------ |
| Outputs:       | nothing, 将其特征附加到另一个意图特征器生成的现有特征向量上  |
| Requires:      | [SpacyNLP](https://rasa.com/docs/rasa/nlu/components/#spacynlp) |
| Description:   | 该特征化器将字符ngram特征附加到特征向量。在训练期间，组件会寻找最常见的字符序列（例如`app`或`ing`）。如果字符序列是否存在于单词序列中，则添加的功能表示布尔标志。注意在此管道之前，还需要另一个意图特征化器！ |
| Configuration: | pipeline: <br />- name: "NGramFeaturizer"   <br />    # Maximum number of ngrams to use when augmenting   <br />    # feature vectors with character ngrams   <br />max_number_of_ngrams: 10 |

###### RegexFeaturizer
| Short:       | 创建正则表达式功能以支持意图和实体分类                       |
| ------------ | ------------------------------------------------------------ |
| Outputs:     | `text_features` and `tokens.pattern`                         |
| Requires:    | nothing                                                      |
| Description: | 在训练期间，正则表达式意图功能化器会创建以训练数据格式定义的正则表达式列表。对于每个正则表达式，将设置一个功能来标记是否在输入中找到了此表达式，然后将其输入意图分类器/实体提取器中以简化分类（假设分类器在训练阶段已获悉，则此设置的功能表示一定的意图）。该`CRFEntityExtractor`组件当前仅支持用于实体提取的正则表达式功能 ！注意pipeline中的此功能化功能之前必须有一个token化功能！ |


##### 意图分类器
###### KeywordIntentClassifier

| Short:          | 简单的关键字匹配意图分类器                                   |
| --------------- | ------------------------------------------------------------ |
| Outputs:        | `intent`                                                     |
| Requires:       | nothing                                                      |
| Output-Example: | `{     "intent": {"name": "greet", "confidence": 0.98343} } ` |
| Description:    | 此分类器主要用作占位符。通过在传递的消息中搜索这些关键字，便能够识别出hello 和 goodbye的意图。 |

###### SklearnIntentClassifier

| Short:          | sklearn意图分类器                                            |
| --------------- | ------------------------------------------------------------ |
| Outputs:        | `intent` and `intent_ranking`                                |
| Requires:       | A featurizer                                                 |
| Output-Example: | `{     "intent": {"name": "greet", "confidence": 0.78343},     "intent_ranking": [         {             "confidence": 0.1485910906220309,             "name": "goodbye"         },         {             "confidence": 0.08161531595656784,             "name": "restaurant_search"         }     ] } ` |
| Description:    | sklearn意图分类器训练了一个线性SVM，该SVM使用网格搜索进行了优化。Spacy意图分类器需要在pipeline中添加特征符。该特征化器创建用于分类的功能。 |
| Configuration:  | 在SVM训练期间，将运行超参数搜索以找到最佳参数集。在配置中，可以指定将尝试使用的参数 |

```
pipeline:
- name: "SklearnIntentClassifier"
  # Specifies the list of regularization values to
  # cross-validate over for C-SVM.
  # This is used with the ``kernel`` hyperparameter in GridSearchCV.
  C: [1, 2, 5, 10, 20, 100]
  # Specifies the kernel to use with C-SVM.
  # This is used with the ``C`` hyperparameter in GridSearchCV.
  kernels: ["linear"]
```

##### Selectors
###### 响应选择

| Short:          | Response Selector                                            |
| --------------- | ------------------------------------------------------------ |
| Outputs:        | A dictionary with key as `direct_response_intent` and value containing `response` and `ranking` |
| Requires:       | A featurizer                                                 |
| Output-Example: | `{     "text": "What is the recommend python version to install?",     "entities": [],     "intent": {"confidence": 0.6485910906220309, "name": "faq"},     "intent_ranking": [         {"confidence": 0.6485910906220309, "name": "faq"},         {"confidence": 0.1416153159565678, "name": "greet"}     ],     "response_selector": {       "faq": {         "response": {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},         "ranking": [             {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},             {"confidence": 0.2134543431, "name": "You can ask me about how to get started"}         ]       }     } } ` |
| Description:    | 响应选择器组件可用于构建响应检索模型，以根据一组候选响应直接预测机器人响应。该模型的预测由[检索动作使用](https://rasa.com/docs/rasa/core/retrieval-actions/#retrieval-actions)。它将用户输入和响应标签嵌入相同的空间，并遵循与完全相同的神经网络架构和优化`EmbeddingIntentClassifier`。响应选择器需要在管道中添加特征符。该特征化器创建用于嵌入的特征。建议使用`CountVectorsFeaturizer`，可以选择在其前面加上`SpacyNLP`。注意如果在预测时间内，一条消息**仅**包含训练中看不见的单词，并且未使用“词汇外”预处理器，则可以有把握地`None`预测出空响应`0.0`。 |
| Configuration:  | 该算法包括所有`EmbeddingIntentClassifier`使用的超参数。此外，该组件还可以配置为针对特定的检索意图训练响应选择器`retrieval_intent`：设置为此响应选择器模型训练的意图的名称。默认`None`在配置中，您可以指定这些参数 |



##### 分词器

###### JiebaTokenizer
| Short:         | Tokenizer using Jieba for Chinese language                   |
| -------------- | ------------------------------------------------------------ |
| Outputs:       | nothing                                                      |
| Requires:      | nothing                                                      |
| Description:   | 使用专用于中文的结巴分词器。对于除中文以外的语言，结巴将作为 `WhitespaceTokenizer`。可用于为MITIE实体提取器token。通过`pip install jieba` |
| Configuration: | 用户的自定义词典文件可以通过以下方式通过文件的特定目录路径自动加载 `dictionary_path`<br />pipeline: <br />- name: "JiebaTokenizer"   <br />    dictionary_path: "path/to/custom/dictionary/dir" |

如果`dictionary_path`为`None`（默认），则将不使用任何自定义词典。

##### Entity Extractors

**CRFEntityExtractor**

| Short:          | CRF实体提取                                                  |
| --------------- | ------------------------------------------------------------ |
| Outputs:        | appends `entities`                                           |
| Requires:       | A tokenizer                                                  |
| Output-Example: | `{     "entities": [{"value":"New York City",                   "start": 20,                   "end": 33,                   "entity": "city",                   "confidence": 0.874,                   "extractor": "CRFEntityExtractor"}] } ` |
| Description:    | 该组件实现条件随机场以进行命名实体识别。可以将CRF视为无向马尔可夫链，其中时间步长是单词，状态是实体类。单词的特征（大写，POS标记等）赋予某些实体类几率，相邻实体标签之间的转换也是如此：然后计算并返回最可能的一组标签。如果使用POS功能（pos或pos2），则必须安装spaCy。 |
| Configuration:  |                                                              |

```
pipeline:
- name: "CRFEntityExtractor"
  # The features are a ``[before, word, after]`` array with
  # before, word, after holding keys about which
  # features to use for each word, for example, ``"title"``
  # in array before will have the feature
  # "is the preceding word in title case?".
  # Available features are:
  # ``low``, ``title``, ``suffix5``, ``suffix3``, ``suffix2``,
  # ``suffix1``, ``pos``, ``pos2``, ``prefix5``, ``prefix2``,
  # ``bias``, ``upper``, ``digit`` and ``pattern``
  features: [["low", "title"], ["bias", "suffix3"], ["upper", "pos", "pos2"]]

  # The flag determines whether to use BILOU tagging or not. BILOU
  # tagging is more rigorous however
  # requires more examples per entity. Rule of thumb: use only
  # if more than 100 examples per entity.
  BILOU_flag: true

  # This is the value given to sklearn_crfcuite.CRF tagger before training.
  max_iterations: 50

  # This is the value given to sklearn_crfcuite.CRF tagger before training.
  # Specifies the L1 regularization coefficient.
  L1_c: 0.1

  # This is the value given to sklearn_crfcuite.CRF tagger before training.
  # Specifies the L2 regularization coefficient.
  L2_c: 0.1
```

[更多Components参考](<https://rasa.com/docs/rasa/nlu/components/#>)


<h1><center>  RASA CORE </center> </h1>
Rasa Core是一个构建人工智能助手的对话引擎，它是Rasa开源框架的一部分。它不是一堆if/else语句，而是通过使用一个经过示例对话训练的机器学习模型来决定下一步做什么。
core主要包含两个内容，stories和domain。
## Stories
Rasa Stories是一种训练数据的形式，用来训练Rasa的对话管理模型。
Stories是用户和人工智能助手之间的对话表示，转换为特定的格式，其中用户输入表示为相应的意图(和必要的实体)，而助手的响应表示为相应的操作名称。
Rasa核心对话系统的一个训练示例称为一个story。
### 格式
以下是Rasa Stories格式的一个对话示例:

```
## greet + location/price + cuisine + num people    <!-- name of the story - just for debugging -->
* greet
   - action_ask_howcanhelp
* inform{"location": "rome", "price": "cheap"}  <!-- user utterance, in format intent{entities} -->
   - action_on_it
   - action_ask_cuisine
* inform{"cuisine": "spanish"}
   - action_ask_numpeople        <!-- action that the bot should execute -->
* inform{"people": "six"}
   - action_ack_dosearch
```
### 构成
- 故事以##开始，例如##story_03248462。您可以将故事命名为任何您喜欢的名称，但是为它们提供描述性的名称对于调试非常有用!
- 故事的结尾用换行符表示，然后用##重新开始新的故事。
- 用户发送的消息以*开头，内容格式为：{"entity1": "value"， "entity2": "value"}。
- bot执行的动作以-开头，包含操作的名称。
- 一个动作返回的事件紧接在该操作之后。例如，如果一个动作返回一个SlotSet事件，它将显示为slot{“slot_name”:“value”}。

### 用户消息
在编写故事时，您不必处理用户发送的消息的特定内容。您可以利用NLU管道的输出，它允许您仅使用意图和实体的组合来引用用户可以发送的所有可能的消息，以表示相同的意思。  
这里包含实体也很重要，因为策略可以根据意图和实体的组合来预测下一步的操作(但是，您可以使用use_entities属性来更改此行为)。 

### 动作
在编写故事时，您将遇到两种类型的操作:话语操作和自定义操作。话语操作是机器人可以回应的硬编码信息，自定义操作则执行自定义代码。  
机器人执行的所有操作(话语和自定义操作)都显示为以-开头的行，后面跟着操作的名称。  
所有的语句都必须以前缀utter_开头，并且必须匹配域中定义的模板的名称。  
对于自定义操作，操作名称是您选择从自定义操作类的name方法返回的字符串。虽然对自定义操作的命名没有限制(与话语不同)，但是这里的最佳实践是在名称前面加上action_。  

### 事件
设置槽位或激活/停用表单等事件必须作为故事的一部分显式地写出来。当自定义操作已经是故事的一部分时，必须单独包含自定义操作返回的事件，这可能看起来有些多余，然而，由于Rasa在训练中不能确定这一事实，这一步是必要的。  
#### 槽位事件
槽事件被写为- Slot{“slot_name”:“value”}。如果此槽设置在自定义操作中，则它将被写入自定义操作事件之后的行中。如果您的自定义操作将槽值重置为None，那么相应的事件将是-slot{“slot_name”:null}。

#### 形态事件
在处理故事的形式时，需要记住三种事件。

- 表单动作事件(例如- restaurant_form)在开始时第一次启动表单时使用，在表单已经激活时恢复表单动作时也会使用。
- 表单激活事件(例如- form{"name": "restaurant_form"})，在第一个表单动作事件之后使用。
- 表单失活事件(例如- form{"name": null})，用于使表单失活。

### 写更少更短的故事
#### Checkpoints
您可以使用>checkpoints来模块化和简化您的训练数据。checkpoints可能有用，但不要过度使用。使用大量checkpoints会使您的示例故事很难理解。如果一个故事块在不同的故事中经常重复，那么使用它们是有意义的，但是没有checkpoints的故事更容易读和写。下面是一个包含checkpoints的示例故事文件(注意，您可以一次附加多个checkpoints)

```
## first story
* greet
   - action_ask_user_question
> check_asked_question

## user affirms question
> check_asked_question
* affirm
  - action_handle_affirmation
> check_handled_affirmation

## user denies question
> check_asked_question
* deny
  - action_handle_denial
> check_handled_denial

## user leaves
> check_handled_denial
> check_handled_affirmation
* goodbye
  - utter_goodbye
```

#### or语句
另一种写短篇故事的方法，或者用同一种方法处理多个意图的方法，是使用or语句。例如，如果您要求用户确认某事，并且您希望以相同的方式对待affirm和thankyou意图。以下故事将在训练时转换为两个故事:

```
## story
...
  - utter_ask_confirm
* affirm OR thankyou
  - action_handle_affirmation
```
就像检查点一样，或语句可能很有用，但是如果您使用了大量检查点或语句，那么最好重新构造您的域或(和)意图。 

**注意**：过度使用这些特性(检查点和或语句)将降低训练速度。

## Domains
Domain定义了人工智能助手所处的世界。它指定了您的机器人应该知道的意图、实体、插槽和操作。另外，它还可以包含您的机器人能够说的内容模板。

### An example of a Domain

```
intents:
  - greet
  - goodbye
  - affirm
  - deny
  - mood_great
  - mood_unhappy
  - bot_challenge

actions:
- utter_greet
- utter_cheer_up
- utter_did_that_help
- utter_happy
- utter_goodbye
- utter_iamabot

templates:
  utter_greet:
  - text: "Hey! How are you?"

  utter_cheer_up:
  - text: "Here is something to cheer you up:"
    image: "https://i.imgur.com/nGF1K8f.jpg"

  utter_did_that_help:
  - text: "Did that help you?"

  utter_happy:
  - text: "Great, carry on!"

  utter_goodbye:
  - text: "Bye"

  utter_iamabot:
  - text: "I am a bot, powered by Rasa
```
NLU模型定义的intents和entities需要包括在域中。
slots中保存着您希望在对话期间跟踪的信息。一个名为risk_level的分类槽的定义如下:

```
slots:
   risk_level:
      type: categorical
      values:
      - low
      - medium
      - high
```
Actions是你的机器人实际上可以做的事情。例如，一个动作可以:
- 回应用户，
- 进行外部API调用，
- 查询数据库
- 或者任何东西!

### 自定义动作和槽位
要引用域中的槽，需要通过它们的模块路径来引用它们。要引用自定义操作，请使用它们的名称。例如，如果您有一个名为my_actions的模块，其中包含一个类MyAwesomeAction，而模块my_slot中包含MyAwesomeSlot，那么您可以将这些行添加到域文件中:
```
actions:
  - my_custom_action
  ...

slots:
  - my_slots.MyAwesomeSlot
```
在本例中，MyAwesomeAction的name函数需要返回my_custom_action

### 话术模板
话术模板是机器人将发送给用户的消息。有两种方式来使用这些模板:  
1. 如果模板的名称以utter_开头，那么可以直接将utterance用作动作。您可以将话语模板添加到域:

```
templates:
  utter_greet:
  - text: "Hey! How are you?"
```
之后，你可以在故事中使用模板作为一个动作:

```
## greet the user
* intent_greet
  - utter_greet
```
当utter_greet作为动作运行时，它将从模板向用户发送消息。
2. 您可以使用模板从使用自定义操作dispatcher: dispatcher.utter_template (“utter_greet”,tracker)生成响应消息。如下所示:
```
from rasa_sdk.actions import Action

class ActionGreet(Action):
  def name(self):
      return 'action_greet'

  def run(self, dispatcher, tracker, domain):
      dispatcher.utter_template("utter_greet", tracker)
      return []
```
### 图片和按钮
在域的yaml文件中定义的模板也可以包含图像和按钮:

```
templates:
  utter_greet:
  - text: "Hey! How are you?"
    buttons:
    - title: "great"
      payload: "great"
    - title: "super sad"
      payload: "super sad"
  utter_cheer_up:
  - text: "Here is something to cheer you up:"
    image: "https://i.imgur.com/nGF1K8f.jpg"
```
### 自定义输出负载

您还可以使用custom:键将任意输出发送到输出通道。注意，由于域是yaml格式的，因此json有效负载应该首先转换为yaml格式。  
例如，尽管日期选择器不是话语模板中定义的参数，因为它们不被大多数通道支持，一个Slack日期选择器可以这样发送:
```
templates:
  utter_take_bet:
  - custom:
      blocks:
      - type: section
        text:
          text: "Make a bet on when the world will end:"
          type: mrkdwn
        accessory:
          type: datepicker
          initial_date: '2019-05-21'
          placeholder:
            type: plain_text
            text: Select a date
```
### 特定通道话术
如果您希望仅将某些话语发送到特定的通道，则可以使用channel: key指定它。该值应该与在通道的OutputChannel类的name()方法中定义的名称匹配。在创建只在特定通道中工作的自定义输出负载时，特定通道的话术特别有用。

```
templates:
  utter_ask_game:
  - text: "Which game would you like to play?"
    channel: "slack"
    custom:
      - # payload for Slack dropdown menu to choose a game
  - text: "Which game would you like to play?"
    buttons:
    - title: "Chess"
      payload: '/inform{"game": "chess"}'
    - title: "Checkers"
      payload: '/inform{"game": "checkers"}'
    - title: "Fortnite"
      payload: '/inform{"game": "fortnite"}'
```
每次机器人寻找话语时，它将首先检查是否有特定于当前连接通道的话术模板。如果有，它只会从这些话术中选择。如果没有，它将从没有定义特定通道的其他话术模板中进行选择。因此，对于每个没有指定通道的话语，最好至少有一个模板，以便您的机器人能够在所有环境中响应。
### 变量
您还可以使用模板中的变量来插入对话期间收集的信息。您可以在自定义python代码中实现这一点，也可以使用自动槽位填充机制。例如，如果你有这样一个模板:

```
templates:
  utter_greet:
  - text: "Hey, {name}. How are you?"
```
Rasa将自动使用name槽中的值填充该变量。
在自定义代码中，您可以使用以下方法检索模板:

```
class ActionCustom(Action):
   def name(self):
      return "action_custom"

   def run(self, dispatcher, tracker, domain):
      # send utter default template to user
      dispatcher.utter_template("utter_default", tracker)
      # ... other code
      return []
```
如果模板包含用{my_variable}表示的变量，您可以通过将它们作为关键字参数传递给utter_template:

```
dispatcher.utter_template("utter_default", tracker, my_variable="my text")
```
### 变体
如果你想随机改变发送给用户的响应，你可以列出多个响应，Rasa会随机选择其中一个，例如:

```
templates:
  utter_greeting:
  - text: "Hey, {name}. How are you?"
  - text: "Hey, {name}. How is your day going?"
```
### 某些意图忽略实体
如果你想要所有的实体在特定的意图下被忽略，你可以像这样在你的域文件中添加use_entities:[]参数到意图:

```
intents:
  - greet:
      use_entities: []
```
要忽略某些实体或显式地只考虑某些实体，可以使用以下语法:

```
intents:
- greet:
    use_entities:
      - name
      - first_name
    ignore_entities:
      - location
      - age
```
这意味着那些意图被排除在外的实体将不作为特征，因此不会影响下一个动作的预测。当您在某些意图中不关心一些实体时，这是很有用的。如果您不使用此参数，所有实体将被作为特征正常使用。

## Responses
如果您希望对话助手响应用户消息，则需要管理这些响应。在机器人的训练数据中，您可以指定机器人应该执行的操作。这些操作可以使用话术模板将消息发送给用户。  

有三种方法来管理这些话术:
1. 包含在域文件中
2. 检索动作响应作为训练数据的一部分
3. 自定义NLG服务来生成响应

### 包含在域文件中
默认格式是将话语包含在域文件中。下面文件可作为所有自定义操作、可用实体、槽位和意图的参考。

```
# all hashtags are comments :)
intents:
 - greet
 - default
 - goodbye
 - affirm
 - thank_you
 - change_bank_details
 - simple
 - hello
 - why
 - next_intent

entities:
 - name

slots:
  name:
    type: text

templates:
  utter_greet:
    - text: "hey there {name}!"  # {name} will be filled by slot (same name) or by custom action
  utter_channel:
    - text: "this is a default channel"
    - text: "you're talking to me on slack!"  # if you define channel-specific utterances, the bot will pick
      channel: "slack"                        # from those when talking on that specific channel
  utter_goodbye:
    - text: "goodbye 😢"   # multiple templates - bot will randomly pick one of them
    - text: "bye bye 😢"
  utter_default:   # utterance sent by action_default_fallback
    - text: "sorry, I didn't get that, can you rephrase it?"

actions:
  - utter_default
  - utter_greet
  - utter_goodbye
```

在这个示例域文件中，templates中包含assistant向用户发送消息的模板。
如果您想更改模板文本或bots响应的其他任何部分，您需要对assistant进行再训练，然后才能进行这些更改。

### 自定义NLG服务来生成响应
对assistant进行再训练以更改文本对于某些工作流来说可能不是最优的。这就是为什么Core也允许你外包响应生成并将其从对话学习中分离出来。  
assistant仍将根据过去的对话学习预测操作和对用户输入作出反应，但它发送给用户的响应是在Rasa核心之外生成的。  
如果assistant向用户发送消息，它将使用POST请求调用外部HTTP服务器。要配置此端点，您需要创建一个端点endpoints.yml。并将其传递给run脚本或server脚本。endpoints.yml端点的内容应该是：

```
nlg:
  url: http://localhost:5055/nlg    # url of the nlg endpoint
  # you can also specify additional parameters, if you need them:
  # headers:
  #   my-custom-header: value
  # token: "my_authentication_token"    # will be passed as a get parameter
  # basic_auth:
  #   username: user
  #   password: pass
# example of redis external tracker store config
tracker_store:
  type: redis
  url: localhost
  port: 6379
  db: 0
  password: password
  record_exp: 30000
# example of mongoDB external tracker store config
#tracker_store:
  #type: mongod
  #url: mongodb://localhost:27017
  #db: rasa
  #user: username
  #password: password
```
然后在启动服务器时将enable-api标志传递给rasa run命令:

```
$ rasa run \
   --enable-api \
   -m examples/babi/models \
   --log-file out.log \
   --endpoints endpoints.yml
```
发送到端点的POST请求内容如下:

```
{
  "tracker": {
    "latest_message": {
      "text": "/greet",
      "intent_ranking": [
        {
          "confidence": 1.0,
          "name": "greet"
        }
      ],
      "intent": {
        "confidence": 1.0,
        "name": "greet"
      },
      "entities": []
    },
    "sender_id": "22ae96a6-85cd-11e8-b1c3-f40f241f6547",
    "paused": false,
    "latest_event_time": 1531397673.293572,
    "slots": {
      "name": null
    },
    "events": [
      {
        "timestamp": 1531397673.291998,
        "event": "action",
        "name": "action_listen"
      },
      {
        "timestamp": 1531397673.293572,
        "parse_data": {
          "text": "/greet",
          "intent_ranking": [
            {
              "confidence": 1.0,
              "name": "greet"
            }
          ],
          "intent": {
            "confidence": 1.0,
            "name": "greet"
          },
          "entities": []
        },
        "event": "user",
        "text": "/greet"
      }
    ]
  },
  "arguments": {},
  "template": "utter_greet",
  "channel": {
    "name": "collector"
  }
}
```
然后，端点需要响应生成的响应:

```
{
    "text": "hey there",
    "buttons": [],
    "image": null,
    "elements": [],
    "attachments": []
}
```
然后，Rasa将使用此响应并将其发送给用户。

## Actions
动作是机器人响应用户输入而运行的操作。在Rasa有四种行为:
1. 话术动作:以utter_开头，向用户发送特定的消息
2. 检索动作:从respond_开始，并发送由检索模型选择的消息
3. 自定义动作:运行任意代码并发送任意数量的消息(或不发送)。
4. 默认动作:例如action_listen, action_restart, action_default_fallback

### 话术动作
要定义一个话术动作(ActionUtterTemplate)，需要将一个话术模板添加到以utter_开头的域文件中:

```
templates:
  utter_my_message:
    - "this is what I want my action to say!"
```
话术动作的名称通常以utter_开始。如果缺少这个前缀，您仍然可以在自定义操作中使用模板，但是模板不能被直接预测为它自己的操作。  
如果使用外部NLG服务，不需要指定域中的模板，但是仍然需要将话语名称添加到域的操作列表中。

### 检索操作
检索动作的设计是为了简化闲聊和简单问题的处理。例如，如果您的助理可以处理100个常见问题和50个不同的闲聊意图，那么您可以使用一个检索操作来涵盖所有这些内容。从对话的角度来看，这些单轮交换可以被平等对待，因此这简化了您的故事。

```
## weather
* ask_weather
   - utter_ask_weather

## introduction
* ask_name
   - utter_introduce_myself

...
```
你可以用一个故事将上面所有意图归类到一个共同的聊天意图下:

```
## chitchat
* chitchat
   - respond_chitchat
```
检索操作使用NLU的响应选择器组件的输出，该组件学习检索模型，从给定用户消息文本的候选响应列表中预测正确的响应。
#### 训练数据
正如名字所暗示的，检索操作学习从候选列表中选择正确的响应。和其他NLU数据一样，你需要在你的NLU文件中包含你的用户要说的话的例子:

```
## intent: chitchat/ask_name
- what's your name
- who are you?
- what are you called?

## intent: chitchat/ask_weather
- how's weather?
- is it sunny where you are?
```
首先，所有这些例子将被合并成一个NLU将要预测的chitchat检索意图。在上面的例子中，所有的检索意图都添加了一个后缀，用于标识assistant的特定响应文本—，如ask_name和ask_weather。  
接下来，将所有检索意图的响应文本作为响应包含在单独的训练数据文件responses.md中。

```
## ask name
* chitchat/ask_name
    - my name is Sara, Rasa's documentation bot!

## ask weather
* chitchat/ask_weather
    - it's always sunny where I live
```
检索模型作为NLU训练管道的一部分单独训练，以选择正确的响应。需要记住的重要一点是，检索模型使用响应消息的文本来选择正确的文本。如果更改这些响应的文本，则必须重新训练检索模型!这是与域文件中的响应模板的关键区别。  
**注意**：包含响应文本的文件必须作为单独的文件存储于传递给训练过程的训练数据目录中，不能是包含NLU其他组件的训练数据文件的一部分。
#### 配置文件
您需要在配置中包含响应选择器(Response Selector )组件。该组件需要一个tokenizer， 一个featurizer和一个意图分类器来对用户消息进行操作，然后才能预测响应，因此这些组件应该放在NLU配置中的ResponseSelector之前。一个例子:

```
language: "en"

pipeline:
- name: "WhitespaceTokenizer"
  intent_split_symbol: "_"
- name: "CountVectorsFeaturizer"
- name: "EmbeddingIntentClassifier"
- name: "ResponseSelector"
```
#### Domain
Rasa使用命名约定将意图名称(如chitchat/ask_name)与检索操作匹配。本例中正确的操作名称是respond_chitchat。必须使用前缀respond_将其标识为检索操作。另一个例子- faq/ask_policy的正确操作名应该是respond_faq，将其包含在您的域中，并将其添加到操作列表中:

```
actions:
  ...
  - respond_chitchat
  - respond_faq
```
确保在聊天意图之后预测检索操作的简单方法是使用映射策略。但是，您也可以在您的故事中包含这个动作。例如，如果您想在处理完闲聊后重复一个问题。

```
## interruption
* search_restaurant
   - utter_ask_cuisine
* chitchat
   - respond_chitchat
   - utter_ask_cuisine
```
#### 多个检索操作
如果assistant同时包含FAQs和chitchat，则可以将它们分离为单独的检索操作，例如有像chitchat/ask_weather和faq/returns_policy这样的意图。Rasa支持添加多个检索操作，如respond_chitchat和respond_returns_policy，为每个意图训练单独的检索模型，你需要在配置中包含一个单独的ResponseSelector组件:

```
language: "en"

pipeline:
- name: "WhitespaceTokenizer"
  intent_split_symbol: "_"
- name: "CountVectorsFeaturizer"
- name: "EmbeddingIntentClassifier"
- name: "ResponseSelector"
  retrieval_intent: chitchat
- name: "ResponseSelector"
  retrieval_intent: faq
```
您仍然可以有两个独立的检索操作，但是通过将一个ResponseSelector组件中retrieval_intent保留为默认值(None)，这两个操作可以共享相同的检索模型。  

到目前为止，在我们的实验中，拥有独立的检索模型对每个检索动作的准确性没有任何影响。为了简单起见，我们建议您在聊天和faq中使用单一的检索模型，如果您得到不同的结果，请在论坛中告诉我们!

#### 解析响应选择器输出
NLU解析后的输出将有一个名为response_selector的属性，其中包含每个响应选择器的输出。每个响应选择器由响应选择器的retrieval_intent参数标识，并存储两个属性：
- 响应:预测响应文本和预测置信度。
- 排名:排名与置信度前10的响应。

结果示例:

```
{
    "text": "What is the recommend python version to install?",
    "entities": [],
    "intent": {"confidence": 0.6485910906220309, "name": "faq"},
    "intent_ranking": [
        {"confidence": 0.6485910906220309, "name": "faq"},
        {"confidence": 0.1416153159565678, "name": "greet"}
    ],
    "response_selector": {
      "faq": {
        "response": {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},
        "ranking": [
            {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},
            {"confidence": 0.2134543431, "name": "You can ask me about how to get started"}
        ]
      }
    }
}
```
如果将特定响应选择器的retrieval_intent参数保留为默认值，则在返回的输出中相应的响应选择器也为默认值。

### 自定义操作
一个可以运行任何你想要运行的代码的动作。自定义操作可以打开灯，将事件添加到日历中，检查用户的银行余额，或者您可以想象的任何其他事情。  
当预测自定义操作时，Rasa将调用您指定的端点。这个端点应该是一个webserver，它响应这个调用，运行代码并可选地返回信息。  
您的操作服务器使用的endpoints.yml示例:

```
action_endpoint:
  url: "http://localhost:5055/webhook"
```
将该端点使用 --endpoints endpoints.yml 传递给脚本。  
您可以使用node.js, .NET, java或其他语言创建一个动作服务器。但是我们提供了一个小型的python SDK，使开发变得更加容易。

#### 用Python编写的自定义操作
对于用python编写的操作，我们有一个方便的SDK为您启动这个操作服务器。
你的动作服务器只需要安装rasa-sdk:

```
pip install rasa-sdk
```
*note*：您不需要为您的操作服务器安装rasa。例如，建议在docker容器中运行Rasa，并为您的动作服务器创建一个单独的容器。在这个单独的容器中，您只需要安装rasa-sdk。  
包含自定义操作的文件称为action.py。如果你安装了rasa，运行这个命令来启动你的动作服务器:

```
rasa run actions
```
如果你没有安装rasa，运行这个命令:

```
python -m rasa_sdk --actions actions
```
在一个餐馆机器人中，如果用户说“给我看看一家墨西哥餐馆”，你的机器人就会执行动作ActionCheckRestaurants，看起来可能是这样的:

```
from rasa_sdk import Action
from rasa_sdk.events import SlotSet

class ActionCheckRestaurants(Action):
   def name(self) -> Text:
      return "action_check_restaurants"

   def run(self,
           dispatcher: CollectingDispatcher,
           tracker: Tracker,
           domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

      cuisine = tracker.get_slot('cuisine')
      q = "select * from restaurants where cuisine='{0}' limit 1".format(cuisine)
      result = db.query(q)

      return [SlotSet("matches", result if result is not None else [])]
```
您应该将动作名称action_check_restaurants添加到域文件中的动作中。操作的run方法接收三个参数。可以使用tracker对象访问槽位的值和用户发送的最新消息，还可以通过dispatcher.utter_template, dispatcher.utter_message或者rasa_sdk.executor.CollectingDispatcher等方法，将消息发送给用户。  

run()方法:

```
Action.run(dispatcher, tracker, domain)
```
**Parameters:**
- dispatcher(CollectingDispatcher)：用于将消息发送给用户。使用dipatcher.utter_message()或rasa_sdk.executor。CollectingDispatcher等方法。
- tracker：当前用户的状态跟踪器。您可以使用tracker.get_slot(slot_name)访问槽值，最近的用户消息是tracker.latest_message，或者其他rasa_sdk.Tracker的属性。
- domain(Dict[Text, Any]) -对话机器人的域

**Returns:**
通过端点返回的事件实例rasa_sdk.events，一个字典。

### 在其他代码中执行操作
Rasa将向您的服务器发送一个HTTP POST请求，其中包含要运行哪个操作的信息。此外，此请求将包含关于对话的所有信息。Action Server展示了详细的API规范。  
作为对来自Rasa的动作调用的响应，您可以修改跟踪器，例如通过设置槽位并将响应发送给用户，所有修改都是使用事件完成的。

### 使用Actions主动接触用户
您可能希望主动接触用户，例如显示长时间运行的后台操作的输出或将外部事件通知用户。  
为此，您可以发布到此端点，指定应该为请求主体中的特定用户运行的操作。使用output_channel查询参数指定应该使用哪个输出通道将助理的响应返回给用户。如果消息是静态的，可以在域文件中使用相应的模板定义一个utter_动作。如果需要更多控制，请在域中添加自定义操作，并在操作服务器中实现所需的步骤。在自定义操作中发送的任何消息都将被转发到指定的输出通道。  
主动接触用户取决于通道的能力，因此并非每个通道都支持。如果您的通道不支持它，可以考虑使用CallbackInput通道向webhook发送消息。  
**注意**：在对话中运行一个动作会改变对话历史并影响助手的下一个预测。如果您不希望发生这种情况，请确保通过将一个ActionReverted事件附加到对话跟踪器的末尾来恢复。
### 默认动作
有八个默认动作:

动作 | 作用
---|---
action_listen | 停止预测更多动作，等待用户输入。
action_restart | 重置整个对话。如果映射策略包含在策略配置中，则可以通过输入/restart在对话期间触发。
action_default_fallback | 撤消最后一条用户消息(就好像用户没有发送它，机器人也没有反应一样)，并发出一条机器人不理解的消息。
action_deactivate_form | 停用活动表单并重置请求的槽位。
action_revert_fallback_events | 恢复TwoStageFallbackPolicy期间发生的事件。
action_default_ask_affirmation | 请用户确认他们的意图。建议使用自定义操作覆盖此默认操作，以获得更有意义的提示。
action_default_ask_rephrase | 要求用户重新表述他们的意图。
action_back | 撤消最后一条用户消息(就好像用户没有发送它，机器人也没有反应一样)。如果策略配置中包含映射策略，则可以通过输入/back在对话期间触发。

所有默认操作都可以被覆盖。为此，请将操作名称添加到域中的操作列表中:

```
actions:
- action_default_ask_affirmation
```
之后，Rasa将调用您的操作端点，并将其视为自定义操作。

## Policies
### 配置策略
类rasa.core.policies决定在对话的每一步中采取什么操作。  
有不同的策略可供选择，您可以在一个rasa.core.agent.Agent中包含多个策略。  
*note* :在每个用户消息之后，代理最多可以预测10个后续操作。要更新这个值，可以将环境变量max_number_of_prediction设置为所需的最大预测数。
您项目中的config.yml文件接受一个policies关键字，您可以使用该关键字定制您的助理使用的策略。在下面的示例中，最后两行说明如何使用自定义策略类并向其传递参数。

```
policies:
  - name: "KerasPolicy"
    featurizer:
    - name: MaxHistoryTrackerFeaturizer
      max_history: 5
      state_featurizer:
        - name: BinarySingleStateFeaturizer
  - name: "MemoizationPolicy"
    max_history: 5
  - name: "FallbackPolicy"
    nlu_threshold: 0.4
    core_threshold: 0.3
    fallback_action_name: "my_fallback_action"
  - name: "path.to.your.policy.class"
    arg1: "..."
```
#### Max History
Rasa核心策略的一个重要超参数是max_history。这控制了模型查看多少对话历史以决定下一步采取什么行动。  
您可以通过在策略配置yaml文件中将max_history传递给策略的Featurizer来设置max_history。  
例如，假设有一个离题的用户消息，表达了超出范围的意图。如果你的机器人连续多次看到这个意图，你可能想告诉用户你可以帮助他们做什么。你的故事可能是这样的:

```
* out_of_scope
   - utter_default
* out_of_scope
   - utter_default
* out_of_scope
   - utter_help_message
```
要让Rasa Core学习这个模式，max_history必须至少为3。  
如果增加max_history，您的模型将变得更大，并且训练将花费更长的时间。如果您有一些将来会影响对话的信息，则应将其存储为插槽。插槽信息始终可用于每个功能块。

#### Data Augmentation
当您训练一个模型时，默认情况下，Rasa Core将通过随机地将故事文件中的故事粘在一起来创建更长的故事。这是因为如果你有这样的故事:

```
# thanks
* thankyou
   - utter_youarewelcome

# bye
* goodbye
   - utter_goodbye
```
实际上，您实际上想自己的策略在不相关时忽略对话历史记录，无论以前发生了什么，都只需以相同的动作进行响应即可。  
您可以使用--augment标识更改此行为。它允许您设置augmentation_factor。augmentation_factor确定在训练期间采样多少个增强故事。增强的故事在训练前会被下采样，因为它们的数量很快就会变得非常大，我们需要限制它。采样的故事数量是augmentation_factor x10。默认情况下，设置为20，即最多增加200个故事。  
--augmentation 0 禁用所有增强行为。基于记忆的策略不受增强的影响(独立于augmentation_factor)，并且将自动忽略所有增强的故事。

### 动作选择
在每一轮对话中，配置中定义的每个策略都会以一定的置信度预测下一个动作。有关每个策略如何做出决策的更多信息，请阅读下面的策略说明。然后，该机器人的下一步操作将由具有最高置信度的预测策略决定。  
在两个策略具有相同置信度的情况下(例如，记忆策略和映射策略总是以0或1的置信度进行预测)，将考虑策略的优先级。Rasa策略有默认的优先级设置，以确保在平局的情况下得到预期的结果。它们看起来是这样的，越高的数字优先级越高:  
5.FormPolicy  
4.FallbackPolicy and TwoStageFallbackPolicy  
3.MemoizationPolicy and AugmentedMemoizationPolicy  
2.MappingPolicy  
1.EmbeddingPolicy, KerasPolicy, and SklearnPolicy  
这个优先级层次结构可以确保，例如，如果有一个带有映射操作的意图，但是NLU的置信度没有超过nlu_threshold，机器人仍然会退回。通常，不建议在每个优先级级别上有多个策略，而同一优先级上的某些策略，例如两个回退策略，严格来说不能同时使用。  
如果您创建自己的策略，请将这些优先级用作确定策略优先级的指南。如果您的策略是机器学习策略，那么它很可能具有优先级1，与Rasa机器学习策略相同。  
**警告** ：所有策略优先级都可以通过配置中的priority参数进行配置，但是我们不建议在特定情况下(例如自定义策略)更改它们。这样做可能会导致意外的和不希望的机器人行为出现。  
### Keras Policy
KerasPolicy使用Keras中实现的神经网络来选择下一步操作。默认的体系结构基于LSTM，但是您可以覆盖KerasPolicy.model_architecture方法来实现您自己的体系结构。

```
def model_architecture(
    self, input_shape: Tuple[int, int], output_shape: Tuple[int, Optional[int]]
) -> tf.keras.models.Sequential:
    """Build a keras model and return a compiled model."""

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Masking,
        LSTM,
        Dense,
        TimeDistributed,
        Activation,
    )

    # Build Model
    model = Sequential()

    # the shape of the y vector of the labels,
    # determines which output from rnn will be used
    # to calculate the loss
    if len(output_shape) == 1:
        # y is (num examples, num features) so
        # only the last output from the rnn is used to
        # calculate the loss
        model.add(Masking(mask_value=-1, input_shape=input_shape))
        model.add(LSTM(self.rnn_size, dropout=0.2))
        model.add(Dense(input_dim=self.rnn_size, units=output_shape[-1]))
    elif len(output_shape) == 2:
        # y is (num examples, max_dialogue_len, num features) so
        # all the outputs from the rnn are used to
        # calculate the loss, therefore a sequence is returned and
        # time distributed layer is used

        # the first value in input_shape is max dialogue_len,
        # it is set to None, to allow dynamic_rnn creation
        # during prediction
        model.add(Masking(mask_value=-1, input_shape=(None, input_shape[1])))
        model.add(LSTM(self.rnn_size, return_sequences=True, dropout=0.2))
        model.add(TimeDistributed(Dense(units=output_shape[-1])))
    else:
        raise ValueError(
            "Cannot construct the model because"
            "length of output_shape = {} "
            "should be 1 or 2."
            "".format(len(output_shape))
        )

    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )

    if obtain_verbosity() > 0:
        model.summary()

    return model
```
模型训练：

```
def train(
    self,
    training_trackers: List[DialogueStateTracker],
    domain: Domain,
    **kwargs: Any
) -> None:

    # set numpy random seed
    np.random.seed(self.random_seed)

    training_data = self.featurize_for_training(training_trackers, domain, **kwargs)
    # noinspection PyPep8Naming
    shuffled_X, shuffled_y = training_data.shuffled_X_y()

    self.graph = tf.Graph()
    with self.graph.as_default():
        # set random seed in tf
        tf.set_random_seed(self.random_seed)
        self.session = tf.compat.v1.Session(config=self._tf_config)

        with self.session.as_default():
            if self.model is None:
                self.model = self.model_architecture(
                    shuffled_X.shape[1:], shuffled_y.shape[1:]
                )

            logger.info(
                "Fitting model with {} total samples and a "
                "validation split of {}"
                "".format(training_data.num_examples(), self.validation_split)
            )

            # filter out kwargs that cannot be passed to fit
            self._train_params = self._get_valid_params(
                self.model.fit, **self._train_params
            )

            self.model.fit(
                shuffled_X,
                shuffled_y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=False,
                verbose=obtain_verbosity(),
                **self._train_params
            )
            # the default parameter for epochs in keras fit is 1
            self.current_epoch = self.defaults.get("epochs", 1)
            logger.info("Done fitting keras policy model")
```
您可以通过覆盖这些方法来实现您选择的模型，或者使用预定义的keras模型初始化KerasPolicy。  
为了对相同的输入获得可重复的训练结果，您可以将KerasPolicy的random_seed属性设置为任意整数。
### Embedding Policy
Transformer version of the Recurrent Embedding Dialogue Policy (REDP)，详见论文[Few-Shot Generalization Across Dialogue Tasks](https://arxiv.org/abs/1811.11707)。  
该策略有一个预先定义的架构，包括以下步骤:  
- 将用户输入(用户意图和实体)、之前的系统动作、每个时间步长的槽和激活表单连接为一个向量，输入到pre-transformer embedding layer；
- 输入transformer；
- 将transformer的输出连接到一个全连接层，得到每个时间步长的对话的embeddings；
- 使用一个全连接层得到每个时间步长的系统动作的embeddings；
- 计算对话embeddings与系统动作embeddings之间的相似度,这一步基于[StarSpace](https://arxiv.org/abs/1709.03856)的想法。  

建议使用state_featurizer=LabelTokenizerSingleStateFeaturizer(…)，
详见[特征方法](https://rasa.com/docs/rasa/api/featurization/)
### Mapping Policy
Mapping Policy可以直接将意图映射到动作，映射关系通过意图的triggers属性来指定，举个例子：

```
intents:
 - ask_is_bot:
     triggers: action_is_bot
```
意图最多只能映射到一个操作。一旦收到触发意图的消息，bot将运行映射的操作。然后，它将监听下一条消息。对于下一个用户消息，将恢复正常的预测。  
如果不希望您的意图-动作映射影响对话历史记录，则映射的动作必须返回UserUtteranceReverted()事件。这将从对话历史记录中删除用户的最新消息及其后发生的任何事件。这意味着您不应该在您的故事中包含意图-动作交互。  
例如，如果用户问“Are you a bot?”在流程进行到一半时，你可能想要在不影响下一次动作预测的情况下回答这个题。这里有一个简单的例子，触发一个机器人的动作，然后恢复互动:

```
class ActionIsBot(Action):
"""Revertible mapped action for utter_is_bot"""

def name(self):
    return "action_is_bot"

def run(self, dispatcher, tracker, domain):
    dispatcher.utter_template("utter_is_bot", tracker)
    return [UserUtteranceReverted()]
```
**注意**：
- 如果您使用MappingPolicy直接预测机器人的话语(例如，trigger: utter_{})，这些交互必须出现在您的故事中，因为在这种情况下没有UserUtteranceReverted()，意图和映射的话语将出现在对话历史中。
- MappingPolicy还负责执行默认动作action_back和action_restart来响应/back和/restart。如果它不包括在您的策略示例中，这些意图将不起作用。

### Memoization Policy
MemoizationPolicy只记录训练数据中的对话。如果训练数据中存在这样的对话，那么它将以confidence 1.0预测下一个动作，否则它将以confidence 0.0预测没有任何动作。

### Fallback Policy
当出现以下情况时，回退策略将调用回退操作:
1. 意图识别的置信度低于nlu_threshold。
2. 排名最高的意图与排名第二的意图之间的置信度差异小于二义性阈值。
3. 所有对话策略预测出的动作的置信度都低于core_threshold。

##### 配置  
阈值和回退操作可以在策略配置文件中作为回退策略的参数进行调整:

```
policies:
  - name: "FallbackPolicy"
    nlu_threshold: 0.3
    ambiguity_threshold: 0.1
    core_threshold: 0.3
    fallback_action_name: 'action_default_fallback'
```
你也可以在你的python代码中配置FallbackPolicy:

```
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.keras_policy import KerasPolicy
from rasa.core.agent import Agent

fallback = FallbackPolicy(fallback_action_name="action_default_fallback",
                          core_threshold=0.3,
                          nlu_threshold=0.3,
                          ambiguity_threshold=0.1)

agent = Agent("domain.yml", policies=[KerasPolicy(), fallback])
```

### Two-Stage Fallback Policy
TwoStageFallbackPolicy通过消除用户输入的歧义，在多个阶段处理NLU的低可信度。
- 如果NLU预测的置信度较低，或者不显著高于排名第二的预测，则要求用户确认意图的分类。
    - 如果他们确认了，故事就会继续下去，就好像这个意图从一开始就以高置信度被识别了一样。
    - 如果他们否认，用户将被要求改述他们的信息。
- 改述
    - 如果对重新措辞的意图的分类具有高置信度，则故事将继续下去，就好像用户从一开始就有此意图一样。
    - 如果重新表达的意图没有被高置信度地分类，则要求用户确认分类的意图。
- 二次确认
    - 如果用户确认了意图，故事就会继续下去，就好像用户从一开始就有这个意图一样。
    - 如果用户拒绝，则原始意图被分类为指定的deny_suggestion_intent_name，并触发最终的回退操作(例如，切换到人工)。
    
##### 配置
要使用TwoStageFallbackPolicy，请在策略配置中包含以下内容：

```
policies:
  - name: TwoStageFallbackPolicy
    nlu_threshold: 0.3
    ambiguity_threshold: 0.1
    core_threshold: 0.3
    fallback_core_action_name: "action_default_fallback"
    fallback_nlu_action_name: "action_default_fallback"
    deny_suggestion_intent_name: "out_of_scope"
```
**注意**：您可以在配置中包含FallbackPolicy或TwoStageFallbackPolicy，但不能两者都包含。

### Form Policy
FormPolicy是处理表单填写的MemoizationPolicy的扩展。一旦调用了FormAction, FormPolicy将不断地预测FormAction，直到表单中所有需要的槽都被填满为止。

## Slots
### 什么是槽(slots)？
槽是机器人的内存。它们充当一个键值存储库，可用于存储用户提供的信(比如家乡城市)以及收集到的关于外部世界的信息(例如数据库查询的结果)。  
大多数时候，你想要一些插槽来影响对话的进展。不同的行为有不同的槽类型。  
例如，如果您的用户提供了他们的家乡城市，那么您可能有一个名为home_city的文本槽。如果用户询问天气，而你不知道他们的家乡，你将不得不询问他们。文本槽只告诉Rasa Core槽是否有值。文本插槽的特定值(如班加罗尔、纽约或香港)没有任何区别。  
如果值本身很重要，那么使用categorical或bool插槽，还有float和list槽。如果您只是想存储一些数据，但又不希望它影响对话的流程，那么就使用unfeaturized槽。
### Rasa如何使用槽？
Policy不直接访问插槽的值，而是接受一个特征表示。如上所述，对于文本槽，值是不相关的。根据是否设置，策略只看到1或0。  
**你应该谨慎选择要使用的槽的类型！**

### 如何设置插槽
你可以在域文件中为每个槽提供一个初始值:

```
slots:
  name:
    type: text
    initial_value: "human"
```
插槽有多种设置方式:
1. 通过NLU设置插槽  
如果您的NLU模型捕获了一个实体，并且您的域包含一个具有相同名称的插槽，那么这个插槽将被自动设置。例如:

```
# story_01
* greet{"name": "Ali"}
  - slot{"name": "Ali"}
  - utter_greet
```
在这种情况下，您不必在故事中包含- slot{}部分，因为它是自动拾取的。  
要禁用特定槽位的这种行为，可以在域文件中将auto_fill属性设置为False:

```
slots:
  name:
    type: text
    auto_fill: False
```

2. 通过点击按钮设置插槽  
您可以使用按钮作为快捷方式。Rasa Core将向RegexInterpreter发送以/开头的消息，RegexInterpreter期望NLU以与story文件相同的格式输入，例如/intent{entities}。例如，如果你让用户通过点击一个按钮来选择一种颜色，按钮的负载可能是/choose{"color": "blue"}和/choose{"color": "red"}。  
你可以在你的域文件中这样指定:

```
utter_ask_color:
- text: "what color would you like?"
  buttons:
  - title: "blue"
    payload: '/choose{"color": "blue"}'
  - title: "red"
    payload: '/choose{"color": "red"}'
```

3. 通过动作设置插槽  
第二个选项是通过在自定义操作中返回事件来设置插槽。在这种情况下，您的故事需要包含插槽。例如，您有一个获取用户属性的自定义操作，以及一个名为account_type的categorical槽。当fetch_profile操作运行时，它返回一个rasa.core.events.SlotSet事件:

```
slots:
   account_type:
      type: categorical
      values:
      - premium
      - basic
```

```
from rasa_sdk.actions import Action
from rasa_sdk.events import SlotSet
import requests

class FetchProfileAction(Action):
    def name(self):
        return "fetch_profile"

    def run(self, dispatcher, tracker, domain):
        url = "http://myprofileurl.com"
        data = requests.get(url).json
        return [SlotSet("account_type", data["account_type"])]
```

```
# story_01
* greet
  - action_fetch_profile
  - slot{"account_type" : "premium"}
  - utter_welcome_premium

# story_02
* greet
  - action_fetch_profile
  - slot{"account_type" : "basic"}
  - utter_welcome_basic
```
在这种情况下，您必须在您的故事中包含- slot{}部分。Rasa Core将学习使用这些信息来决定采取正确的行动(在本例中是utter_welcome_premium或utter_welcome_basic)。  
**注意**：如果你用手写故事，很容易忘记插槽。我们强烈建议您使用表单的交互式学习来构建这些故事，而不是手动编写。

### Slot Types
#### Text Slot
- 用途：用户偏好，您只关心它们是否已被指定。
- 举例：
```
slots:
   cuisine:
      type: text
```
- 描述：
如果设置了任何值，则槽的特征值被设置为1，否则特征值将被设置为0(未设置任何值)。
#### Boolean Slot
- 用途：True or False
- 举例：

```
slots:
   is_authenticated:
      type: bool
```
- 描述：检查插槽是否设置和是否为真
#### Categorical Slot
- 用途：可以从N个值中选择一个
- 举例：

```
slots:
   risk_level:
      type: categorical
      values:
      - low
      - medium
      - high
```
- 描述：创建一个one-hot编码，描述匹配的值。
#### Float Slot
- 用途：连续值
- 举例：

```
slots:
   temperature:
      type: float
      min_value: -100.0
      max_value:  100.0
```
- 默认值：max_value=1.0, min_value=0.0
- 描述：min_value以下的所有值都将被视为min_value, max_value以上的值也将被视为min_value。因此，如果max_value设置为1，槽值2和3.5在特征化方面没有区别(例如，两个值都会以相同的方式影响对话，并且模型无法学会区分它们)。
#### List Slot
- 用途：列表
- 举例：

```
slots:
   shopping_items:
      type: list
```
- 描述：如果设置了列表的值，列表不为空，则此槽的特征为1。如果没有设置值，或者列表为空，则该特征为0。**存储在槽中的列表的长度不影响对话**。
#### Unfeaturized Slot
- 用途：存储数据但不想影响对话流
- 举例：

```
slots:
   internal_user_id:
      type: unfeaturized
```
- 描述：这个槽不会有任何的特征，因此它的值不会影响对话流，并且在预测机器人的下一个动作时被忽略。
### Custom Slot Types
也许你的餐厅预订系统最多只能处理6个人的预订。在本例中，您希望插槽的值影响下一个选定的操作(而不仅仅是它是否被指定)。您可以通过定义一个定制的slot类来实现这一点。  
在下面的代码中，我们定义了一个名为NumberOfPeopleSlot的槽类。特征化定义了如何将这个槽的值转换为向量，以使我们的机器学习模型能够处理。我们的槽有三个可能的“值”，我们可以用长度为2的向量来表示它们。

值 | 描述
---|---
(0,0)) | 未设置
(1,0) | 1-6之间
(0,1) | 大于6

```
from rasa.core.slots import Slot

class NumberOfPeopleSlot(Slot):

    def feature_dimensionality(self):
        return 2

    def as_feature(self):
        r = [0.0] * self.feature_dimensionality()
        if self.value:
            if self.value <= 6:
                r[0] = 1.0
            else:
                r[1] = 1.0
        return r
```
现在我们还需要一些训练故事，这样Rasa Core可以从中学习如何处理不同的情况:

```
# story1
...
* inform{"people": "3"}
  - action_book_table
...
# story2
* inform{"people": "9"}
  - action_explain_table_limit
```
## Forms
最常见的会话模式之一是从用户那里收集一些信息，以便进行某些操作(预订餐馆、调用API、搜索数据库等)。这也叫做**填槽**。  
如果您需要在一行中收集多个信息片段，我们建议您创建一个FormAction。这是一个单独的操作，它包含循环所需插槽并向用户询问此信息的逻辑。在Rasa Core的examples/formbot目录中有一个使用表单的完整示例。  
定义表单时，需要将其添加到域文件中。如果你的表单的名字是restaurant_form，你的域名应该是这样的:

```
forms:
  - restaurant_form
actions:
  ...
```
### 配置文件
要使用表单，还需要在策略配置文件中包含FormPolicy。例如:

```
policies:
  - name: "FormPolicy"
```
### Form Basics
使用FormAction，您可以用一个故事描述所有的happy path。我们所说的“happy path”，指的是无论何时你向用户询问一些信息，他们都会回复你所询问的信息。  
如果我们以餐厅机器人为例，下面这个故事描述了所有的happy path。

```
## happy path
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - form{"name": null}
```
在这个故事中，用户的意图是request_restaurant，后面跟着表单action restaurant_form。对于表单{“name”:“restaurant_form”}，表单被激活，对于表单{“name”:null}，表单再次被停用。如处理unhappy path一节所示，当表单仍然处于活动状态时，机器人可以在表单之外执行任何类型的操作。在“happy path”上，用户协作良好，系统正确地理解用户输入，表单正在无中断地填充所有请求的插槽。  
FormAction只会请求尚未设置的插槽。如果一个用户在会话开始说“I’d like a vegetarian Chinese restaurant for 8 people”，那么他们将不会被问及cuisine和num_people插槽。  
请注意，要使这个故事起作用，您的插槽应该是unfeaturized的。如果这些插槽中的任何一个是特征化的，您的故事需要包含slot{}事件来显示这些插槽被设置。在这种情况下，创建有效的故事的最简单方法是使用交互式学习。  
在上面的故事中，restaurant_form是表单动作的名称。下面是一个例子。你需要定义三个方法:  
- name:操作的名称
- required_slot:提交的方法工作时需要填充的槽的列表。
- submit:当所有槽位都被填充后，应该执行的操作。

```
def name(self) -> Text:
    """Unique identifier of the form"""

    return "restaurant_form"
```

```
@staticmethod
def required_slots(tracker: Tracker) -> List[Text]:
    """A list of required slots that the form has to fill"""

    return ["cuisine", "num_people", "outdoor_seating", "preferences", "feedback"]
```

```
def submit(
    self,
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: Dict[Text, Any],
) -> List[Dict]:
    """Define what the form has to do
        after all required slots are filled"""

    # utter submit template
    dispatcher.utter_template("utter_submit", tracker)
    return []
```
一旦表单动作第一次被调用，表单就会被激活，FormPolicy就会介入。FormPolicy非常简单，总是预测表单动作。有关如何处理未预期的用户输入，请参阅处理unhappy path一节。  
每次表单操作被调用时,它会要求用户输入required_slots中未设置的下一个槽,这是通过寻找一个称为utter_ask_ {slot_name}的模板来完成的,所以你需要在域文件中定义这些插槽。  
一旦所有的槽被填满，就会调用submit()方法，您可以使用收集到的信息为用户做一些事情，例如查询餐馆API。如果不希望表单在最后执行任何操作，只需使用return[]作为提交方法。在调用submit方法之后，表单将被停用，您的核心模型中的其他策略将用于预测下一步操作。
### 自定义槽映射
如果不定义槽映射，则插槽将仅由与从用户输入中插槽名称相同的实体填充。有些插槽，比如cuisine，可以使用单个实体来填充，但FormAction也可以支持yes/no问题和自由文本输入。slot_mappings方法定义了如何从用户响应中提取槽值。  
下面是餐厅机器人的一个例子:

```
def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
    """A dictionary to map required slots to
        - an extracted entity
        - intent: value pairs
        - a whole message
        or a list of them, where a first match will be picked"""

    return {
        "cuisine": self.from_entity(entity="cuisine", not_intent="chitchat"),
        "num_people": [
            self.from_entity(
                entity="num_people", intent=["inform", "request_restaurant"]
            ),
            self.from_entity(entity="number"),
        ],
        "outdoor_seating": [
            self.from_entity(entity="seating"),
            self.from_intent(intent="affirm", value=True),
            self.from_intent(intent="deny", value=False),
        ],
        "preferences": [
            self.from_intent(intent="deny", value="no additional preferences"),
            self.from_text(not_intent="affirm"),
        ],
        "feedback": [self.from_entity(entity="feedback"), self.from_text()],
    }
```
预定义的功能如下:
- 如果intent_name是None或者用户意图是intent_name，self.from_entity(entity=entity_name, intent=intent_name)将查找名为entity_name的实体来填充槽slot_name。
- 如果用户意图是intent_name，则self.from_intent(intent=intent_name, value=value)将用值填充槽slot_name。要创建一个布尔槽，请查看上面的outdoor_sitting定义。注意:插槽不会被触发表单动作的用户消息填充。使用self.from_trigger_intent如下。
- 如果表单被用户意图 intent_name触发，self.from_trigger_intent(intent=intent_name, value=value)将用值填充 slot_name。
- 如果intent_name是None或者用户意图是intent_name，则self.from_text(intent=intent_name)将使用下一个用户语句来填充文本槽slot_name。
- 如果您希望使用它们的组合，请像上面的示例那样以列表的形式提供它们。
### 验证用户输入
从用户输入提取槽值后，表单将尝试验证槽值。注意，在默认情况下，只有在用户输入之后立即执行表单操作时才会进行验证。这可以在Rasa SDK中FormAction类的_validate_if_required()函数中更改。在初始激活表单之前填充的所有必需的插槽也在激活时进行验证。  
默认情况下，验证只检查请求的槽是否成功地从槽映射中提取出来。如果您想添加自定义验证，例如针对数据库检查一个值，您可以通过编写名为validate_{slot-name}的帮助验证函数来实现。  
下面是一个示例validate_cuisine()，它检查提取的cuisine槽是否属于支持的cuisines列表。

```
 @staticmethod
    def cuisine_db() -> List[Text]:
        """Database of supported cuisines"""

        return [
            "caribbean",
            "chinese",
            "french",
            "greek",
            "indian",
            "italian",
            "mexican",
        ]
```

```
def validate_cuisine(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate cuisine value."""

        if value.lower() in self.cuisine_db():
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"cuisine": value}
        else:
            dispatcher.utter_template("utter_wrong_cuisine", tracker)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"cuisine": None}
```
由于helper验证函数返回要设置的槽名和值的字典，因此可以设置比helper验证方法中更多的槽。但是，您需要确保这些额外的槽值是有效的。  
您还可以通过返回self.deactivate()来在这个验证步骤中直接停用表单(如果槽中填充了您确信无法处理的内容)。  
如果没有从用户的话语中提取任何需要的插槽，ActionExecutionRejection错误将被抛出，这意味着动作执行被拒绝，因此Core将返回到其他策略来预测另一个动作。  
### 处理unhappy paths
当然，您的用户并不总是响应您向他们询问的信息。通常情况下，用户会问问题，闲聊，改变主意，或者偏离正轨。处理表单的方式是，如果用户没有提供所请求的信息，表单将引发ActionExecutionRejection。您需要处理可能在您的故事中导致ActionExecutionRejection错误的事件。例如，如果你希望你的用户和你的机器人聊天，你可以添加这样一个故事:

```
## chitchat
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* chitchat
    - utter_chitchat
    - restaurant_form
    - form{"name": null}
```
在某些情况下，用户可能在表单操作过程中改变主意，决定不再继续最初的请求。在这种情况下，助理应该停止请求所请求的插槽。您可以使用默认操作action_deactivate_form优雅地处理这种情况，该操作将禁用表单并重置请求的槽。下面是这类对话的一个例子:

```
## chitchat
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
* stop
    - utter_ask_continue
* deny
    - action_deactivate_form
    - form{"name": null}
```
强烈建议您使用交互式学习构建这些故事。如果你手写这些故事，你可能会错过重要的东西。详情请阅读使用表单进行互动学习一节。  
### requested_slot
requested_slot是作为unfeaturized slot自动添加到域中的。如果您想要使它特征化，您需要将它作为一个分类槽添加到您的域文件中。如果您希望根据用户当前询问的槽位以不同的方式处理unhappy paths，那么您可能需要这样做。例如，假设您的用户用另一个问题来回答机器人的一个问题，比如“为什么需要知道这个?”，对这个explain意图的回应取决于我们在故事中的位置。在餐馆的例子中，你的故事应该是这样的:

```
## explain cuisine slot
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - slot{"requested_slot": "cuisine"}
* explain
    - utter_explain_cuisine
    - restaurant_form
    - slot{"cuisine": "greek"}
    ( ... all other slots the form set ... )
    - form{"name": null}

## explain num_people slot
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - slot{"requested_slot": "num_people"}
* explain
    - utter_explain_num_people
    - restaurant_form
    - slot{"cuisine": "greek"}
    ( ... all other slots the form set ... )
    - form{"name": null}
```
再次强烈建议您使用交互式学习构建这些故事。如果你手写这些故事，你可能会错过重要的东西。详情请阅读使用表单进行互动学习一节。 
### 处理条件槽逻辑
许多表单需要比请求字段列表更多的逻辑。例如，如果有人要求希腊菜，你可能想问他们是否在寻找一个有露天座位的地方。  
您可以通过将一些逻辑写入required_slot()方法来实现这一点，例如:

```
@staticmethod
def required_slots(tracker) -> List[Text]:
   """A list of required slots that the form has to fill"""

   if tracker.get_slot('cuisine') == 'greek':
     return ["cuisine", "num_people", "outdoor_seating",
             "preferences", "feedback"]
   else:
     return ["cuisine", "num_people",
             "preferences", "feedback"]
```
这种机制非常普遍，您可以使用它在表单中构建许多不同类型的逻辑。
### Debugging
首先要做的是使用debug标志运行您的bot，有关详细信息，请参阅[命令行接口](https://rasa.com/docs/rasa/user-guide/command-line-interface/#command-line-interface)。如果你刚刚开始，你可能只有几个手写的故事。这是一个很好的起点，但是您应该尽快将您的机器人交给真人进行测试。Rasa核心的指导原则之一是:
**从真实的对话中学习比设计假想的对话更重要**
因此，在将您的故事交给测试人员之前，不要试图覆盖所有的可能性。真正的用户行为总是会让你惊讶!
## Interactive Learning
这个页面展示了如何在命令行上使用交互式学习。  
在交互式学习模式中，你在与机器人交谈时向它提供反馈。这是一种强大的方式来探索你的机器人可以做什么，以及最简单的方式来修复它所犯的任何错误。基于机器学习的对话的一个优点是，当你的机器人还不知道如何做某事时，你可以直接教它!有些人称之为[Software 2.0](https://medium.com/@karpathy/software-2-0-a64152b37c35)。  
Rasa X提供了一个用于交互式学习的UI，您可以使用任何用户对话作为起点。请参阅Rasa X文档中的[注释对话](https://rasa.com/docs/rasa-x/annotate-conversations/)。
### 运行交互式学习
运行以下命令开始交互式学习:

```
rasa run actions --actions actions&

rasa interactive \
  -m models/20190515-135859.tar.gz \
  --endpoints endpoints.yml
```
第一个命令启动操作服务器。  
第二个命令启动交互式学习模式。  
在交互模式下，Rasa会要求你确认NLU和Core做出的每一个预测。这里有一个例子:  

```
Bot loaded. Type a message and press enter (use '/stop' to exit).

? Next user input:  hello

? Is the NLU classification for 'hello' with intent 'hello' correct?  Yes

------
Chat History

 #    Bot                        You
────────────────────────────────────────────
 1    action_listen
────────────────────────────────────────────
 2                                    hello
                         intent: hello 1.00
------

? The bot wants to run 'utter_greet', correct?  (Y/n)
```
聊天记录和插槽值被打印到屏幕上，这应该是您决定下一步正确操作所需的所有信息。  
在这种情况下，机器人选择了正确的动作(utter_greet)，所以我们输入y，然后再次输入y，因为action_listen是问候后的正确动作。我们继续这个循环，与机器人聊天，直到机器人选择了错误的动作。  
### 提供错误反馈
对于本例，我们将使用concertbot示例，因此请确保您拥有它的域和数据。你可以从我们的[github repo](https://github.com/RasaHQ/rasa/tree/master/examples/concertbot)下载数据。  
如果您询问/search_concert，机器人应该建议action_search_concert，然后是action_listen(策略选择下一个动作时的置信程度将显示在动作名称旁边)。现在让我们输入/compare_reviews作为下一个用户消息。机器人可能会从两种可能性中选择错误的一种(取决于训练运行，它也可能是正确的):

```
------
Chat History

 #    Bot                                           You
───────────────────────────────────────────────────────────────
 1    action_listen
───────────────────────────────────────────────────────────────
 2                                            /search_concerts
                                  intent: search_concerts 1.00
───────────────────────────────────────────────────────────────
 3    action_search_concerts 0.72
      action_listen 0.78
───────────────────────────────────────────────────────────────
 4                                            /compare_reviews
                                  intent: compare_reviews 1.00


Current slots:
  concerts: None, venues: None

------
? The bot wants to run 'action_show_venue_reviews', correct?  No
```
现在我们输入n，因为它选择了错误的操作，我们会得到一个新的提示，要求输入正确的操作。这也显示了模型分配给每个动作的概率:

```
? What is the next action of the bot?  (Use arrow keys)
 ❯ 0.53 action_show_venue_reviews
   0.46 action_show_concert_reviews
   0.00 utter_goodbye
   0.00 action_search_concerts
   0.00 utter_greet
   0.00 action_search_venues
   0.00 action_listen
   0.00 utter_youarewelcome
   0.00 utter_default
   0.00 action_default_fallback
   0.00 action_restart
```
在这种情况下，机器人的动作应该是action_show_concert_reviews(而不是enue_reviews!)，所以我们选择了这个动作。  
现在，只要我们想要创造一个更长的对话，我们就可以一直和机器人对话。在任何时候，你可以按Ctrl-C，机器人将为你提供退出选项。您可以将新创建的故事和NLU数据写入文件。如果你在提供反馈时犯了错误，你也可以退回一步。  
确保将转储的故事和NLU示例与您的原始训练数据结合起来，以供下一次训练使用。
### 对话可视化
在交互学习过程中，Rasa会根据训练数据绘制出当前的对话和一些类似的对话，以帮助你跟踪自己的位置。  
一旦开始了交互式学习，就可以通过http://localhost:5005/visualization.html查看可视化结果。  
要跳过可视化，运行rasa interactive --skip-visualization。  
![](https://rasa.com/docs/rasa/_images/interactive_learning_graph.gif)
### 表单交互学习
如果您正在使用FormAction，那么在使用交互式学习时需要记住一些额外的东西。 
#### form:前缀
表单逻辑由FormAction类描述，而不是由story描述。机器学习策略不应该必须学习这种行为，而且如果以后更改表单操作(例如添加或删除一个必需的槽)，也不应该感到困惑。当您使用交互式学习来生成包含表单的故事时，表单处理的对话步骤将获得一个form:前缀。这告诉Rasa Core在训练其他策略时忽略这些步骤。你在这里没有什么特别要做的，所有形式的happy paths仍然被基本的故事所覆盖。  
下面是一个例子：

```
* request_restaurant
    - restaurant_form
    - form{"name": "restaurant_form"}
    - slot{"requested_slot": "cuisine"}
* form: inform{"cuisine": "mexican"}
    - slot{"cuisine": "mexican"}
    - form: restaurant_form
    - slot{"cuisine": "mexican"}
    - slot{"requested_slot": "num_people"}
* form: inform{"number": "2"}
    - form: restaurant_form
    - slot{"num_people": "2"}
    - form{"name": null}
    - slot{"requested_slot": null}
    - utter_slots_values
```
#### 输入验证
每当用户使用请求的插槽或任何必需的插槽以外的内容进行响应时，您将被询问是否希望表单操作尝试在返回表单时从用户的消息中提取插槽。这是最好的解释与例子:

```
 7    restaurant_form 1.00
      slot{"num_people": "3"}
      slot{"requested_slot": "outdoor_seating"}
      do you want to sit outside?
      action_listen 1.00
─────────────────────────────────────────────────────────────────────────────────────
 8                                                                             /stop
                                                                   intent: stop 1.00
─────────────────────────────────────────────────────────────────────────────────────
 9    utter_ask_continue 1.00
      do you want to continue?
      action_listen 1.00
─────────────────────────────────────────────────────────────────────────────────────
 10                                                                          /affirm
                                                                 intent: affirm 1.00


Current slots:
    cuisine: greek, feedback: None, num_people: 3, outdoor_seating: None,
  preferences: None, requested_slot: outdoor_seating

------
2018-11-05 21:36:53 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'default'
? The bot wants to run 'restaurant_form', correct?  Yes
2018-11-05 21:37:08 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'default'
? Should 'restaurant_form' validate user input to fill the slot 'outdoor_seating'?  (Y/n)
```
这里，用户要求停止表单，机器人询问用户是否确定不想继续。用户说他们想继续(确认意图)。这里outdoor_setting有一个from_intent槽映射(将/affirm intent映射为True)，所以这个用户输入可以用来填充那个槽。然而，在这种情况下，用户只是回应“do you want to continue?”问题，所以你选择n，用户输入不应该被验证。然后机器人会再次询问outdoor_seating槽。  
**警告**
如果有一个冲突的故事出现在你的训练数据中,例如你选择验证输入(这意味着它将带有forms:前缀),但你的故事文件包含另一个相同的故事,而你不需要验证输入(没有forms:前缀),这时您将需要删除这个冲突的故事。当这种情况发生时，会出现一个警告提示，提醒您这样做:
WARNING: FormPolicy predicted no form validation based on previous training stories. Make sure to remove contradictory stories from training data
您可以在删除冲突故事后按回车键继续互动学习。
## Knowledge Base Actions
**注意**
这个特征是实验性的。我们引入了实验性的特征来获得社区的反馈，所以我们鼓励您尝试一下!但是，将来可能会更改或删除该功能。如果你有反馈(积极的或消极的)，请在[论坛](https://forum.rasa.com/?_ga=2.240825475.1039339409.1571564377-307585668.1569235451)上与我们分享。  
知识库操作使您能够处理以下类型的对话:  
![](https://rasa.com/docs/rasa/_images/knowledge-base-example.png)
在人工智能会话中，一个常见的问题是，用户不仅使用名称来指代某些对象，而且还使用“第一个”或“它”等引用术语。我们需要跟踪显示的信息，以将这些提及解析为正确的对象。  
此外，用户可能希望在对话期间获得关于对象的详细信息——例如，餐馆是否有户外座位，或者有多贵。为了响应这些用户请求，需要了解餐馆领域的知识。由于信息可能会改变，硬编码信息并不是解决方案。  
为了应对上述挑战，Rasa可以与知识库相结合。要使用这种集成，您可以创建一个自定义操作，它继承自ActionQueryKnowledgeBase，这是一个预先编写好的自定义操作，它包含了查询对象及其属性知识库的逻辑。  
你可以在example/knowledgebasebot([knowledge base bot](https://github.com/RasaHQ/rasa/blob/master/examples/knowledgebasebot/))中找到一个完整的例子，下面是实现这个自定义操作的说明。

### 使用ActionQueryKnowledgeBase
#### 创建知识库
用于回答用户请求的数据将存储在知识库中。知识库可用于存储复杂的数据结构。我们建议你从使用InMemoryKnowledgeBase开始。一旦开始处理大量数据，就可以切换到自定义知识库。  
要初始化InMemoryKnowledgeBase，需要在json文件中提供数据。下面的示例包含关于餐馆和酒店的数据。json结构应该包含每个对象类型的键，例如“餐厅”和“酒店”。每个对象类型都映射到一个对象列表——这里我们有一个包含3个餐馆和3个酒店的列表。

```
{
    "restaurant": [
        {
            "id": 0,
            "name": "Donath",
            "cuisine": "Italian",
            "outside-seating": true,
            "price-range": "mid-range"
        },
        {
            "id": 1,
            "name": "Berlin Burrito Company",
            "cuisine": "Mexican",
            "outside-seating": false,
            "price-range": "cheap"
        },
        {
            "id": 2,
            "name": "I due forni",
            "cuisine": "Italian",
            "outside-seating": true,
            "price-range": "mid-range"
        }
    ],
    "hotel": [
        {
            "id": 0,
            "name": "Hilton",
            "price-range": "expensive",
            "breakfast-included": true,
            "city": "Berlin",
            "free-wifi": true,
            "star-rating": 5,
            "swimming-pool": true
        },
        {
            "id": 1,
            "name": "Hilton",
            "price-range": "expensive",
            "breakfast-included": true,
            "city": "Frankfurt am Main",
            "free-wifi": true,
            "star-rating": 4,
            "swimming-pool": false
        },
        {
            "id": 2,
            "name": "B&B",
            "price-range": "mid-range",
            "breakfast-included": false,
            "city": "Berlin",
            "free-wifi": false,
            "star-rating": 1,
            "swimming-pool": false
        },
    ]
}
```
在json文件中定义数据之后，例如，data.json，您便可以使用这个数据文件来创建您的InMemoryKnowledgeBase，它将被传递给查询知识库的操作。  
知识库中的每个对象都应该至少有“name”和“id”字段，以便使用默认实现。如果没有，你必须自定义自己的InMemoryKnowledgeBase。
#### 定义NLU数据
在本节中:
- 我们将引入一个新的意图，query_knowledge_base
- 我们将注释mention实体，这样我们的模型就能检测到间接提到的对象，比如“第一个”
- 我们将广泛使用[同义词](https://rasa.com/docs/rasa/nlu/training-data-format/#entity-synonyms)

为了让机器人理解用户想要从知识库中检索信息，您需要定义一个新的意图。我们将其命名为query_knowledge_base。  
我们可以将ActionQueryKnowledgeBase能够处理的请求分为两类:(1)用户想要获取特定类型对象的列表，(2)用户想要了解对象的某个属性。意图应该包含这两个请求的许多变体:

```
## intent:query_knowledge_base
- what [restaurants](object_type:restaurant) can you recommend?
- list some [restaurants](object_type:restaurant)
- can you name some [restaurants](object_type:restaurant) please?
- can you show me some [restaurant](object_type:restaurant) options
- list [German](cuisine) [restaurants](object_type:restaurant)
- do you have any [mexican](cuisine) [restaurants](object_type:restaurant)?
- do you know the [price range](attribute:price-range) of [that one](mention)?
- what [cuisine](attribute) is [it](mention)?
- do you know what [cuisine](attribute) the [last one](mention:LAST) has?
- does the [first one](mention:1) have [outside seating](attribute:outside-seating)?
- what is the [price range](attribute:price-range) of [Berlin Burrito Company](restaurant)?
- what about [I due forni](restaurant)?
- can you tell me the [price range](attribute) of [that restaurant](mention)?
- what [cuisine](attribute) do [they](mention) have?
 ...
```
上面的例子只是展示了与餐馆领域相关的例子。您应该将知识库中存在的每种对象类型的示例添加到相同的query_knowledge_base意图中。  
除了为每种查询类型添加各种训练示例外，您还需要在训练示例中指定并注释以下实体:  
- object_type:每当一个训练示例从您的知识库引用特定对象类型时，对象类型都应该标记为实体。使用[同义词](https://rasa.com/docs/rasa/nlu/training-data-format/#entity-synonyms)来映射，例如，restaurants -> restaurant，正确的对象类型在知识库中作为键列出。
- mention:如果用户通过“第一个”、“那个”或“它”引用某个对象，那么应该将这些术语标记为mention。我们也使用同义词来映射一些提到的符号。
- attribute:在您的知识库中定义的所有属性名称都应该被标识为NLU数据中的属性。同样，使用同义词将属性名的变体映射到知识库中使用的变体。  
请记住将这些实体添加到您的域文件(作为实体和插槽):

```
entities:
  - object_type
  - mention
  - attribute

slots:
  object_type:
    type: unfeaturized
  mention:
    type: unfeaturized
  attribute:
    type: unfeaturized
```
#### 创建查询知识库的操作
要创建自己的知识库动作，需要继承ActionQueryKnowledgeBase并将知识库传递给ActionQueryKnowledgeBase的构造函数。

```
class MyKnowledgeBaseAction(ActionQueryKnowledgeBase):
    def __init__(self):
        knowledge_base = InMemoryKnowledgeBase("data.json")
        super().__init__(knowledge_base)
```
无论何时创建ActionQueryKnowledgeBase，都需要将一个知识库传递给构造函数。它可以是一个不需要记忆的知识库，也可以是你自己实现的知识库。您只能从一个知识库提取信息，因为不支持同时使用多个知识库。  
这就是这个操作的全部代码!动作的名称是action_query_knowledge_base。不要忘记添加到您的域文件:

```
actions:
- action_query_knowledge_base
```
**注意** 如果您覆盖了默认的动作名action_query_knowledge_base，那么您需要将以下三个unfeaturized槽添加到您的域文件中:knowledge_base_objects、knowledge_base_last_object和knowledge_base_last_object_type。这些插槽由ActionQueryKnowledgeBase内部使用。如果您保留默认的操作名称，这些插槽将自动为您添加。  
您还需要确保将一个故事添加到您的故事文件中，其中包括意图query_knowledge_base和动作action_query_knowledge_base。例如:

```
## Happy Path
* greet
  - utter_greet
* query_knowledge_base
  - action_query_knowledge_base
* goodbye
  - utter_goodbye
```
您需要做的最后一件事是在域文件中定义模板utter_ask_rephrase。如果操作不知道如何处理用户的请求，它将使用此模板要求用户重新措辞。例如，添加以下模板到您的域文件:

```
utter_ask_rephrase:
- text: "Sorry, I'm not sure I understand. Could you rephrase it?"
- text: "Could you please rephrase your message? I didn't quite get that."
```
在添加了所有相关部分之后，操作现在就可以查询知识库了。
### 如何运作
ActionQueryKnowledgeBase既会查看请求中获取的实体，也会查看之前设置的槽来决定查询什么。  
#### 从知识库中查询对象
为了在知识库中查询任何类型的对象，用户的请求需要包含对象类型。让我们来看一个例子:

```
Can you please name some restaurants?
```
这个问题包括感兴趣的对象类型:“restaurant”。机器人需要利用这个实体来形成一个查询——否则操作将不知道用户感兴趣的对象是什么。  
当用户这样说:

```
What Italian restaurant options in Berlin do I have?
```
用户希望获得(1)拥有意大利美食和(2)位于柏林的的餐馆列表。如果NER在用户请求中检测到这些属性，则操作将使用这些属性来筛选知识库中找到的餐馆。  
为了让机器人检测到这些属性，你需要在NLU数据中将“意大利”和“柏林”标记为实体:

```
What [Italian](cuisine) [restaurant](object_type) options in [Berlin](city) do I have?.
```
属性的名称“cuisine”和“city”应该与知识库中使用的名称相同。您还需要将这些作为实体和插槽添加到域文件中。
#### 从知识库中查询对象属性
如果用户希望获得有关对象的特定信息，则请求应该同时包含相关对象和属性。例如，如果用户这样问:

```
What is the cuisine of Berlin Burrito Company?
```
用户希望获得“Berlin Burrito Company”(兴趣对象)的“cuisine”(兴趣属性)。  
感兴趣的属性和对象应该标记为NLU训练数据中的实体:

```
What is the [cuisine](attribute) of [Berlin Burrito Company](restaurant)?
```
确保将对象类型“restaurant”作为实体和槽添加到域文件中。
#### 处理Mentions
按照上面的例子，用户可能并不总是通过他们的名字来引用餐馆。用户可以通过其名称来引用感兴趣的对象，例如“Berlin Burrito Company”，或者他们可以通过引用来指之前列出的对象，例如:

```
What is the cuisine of the second restaurant you mentioned?
```
我们的操作能够将这些mentions解析为知识库中的实际对象。更具体地说，它可以解决两种mentions类型:(1)序数mentions，如“第一个”，(2)mentions如“它”或“那个”。
##### 序数mentions
当用户通过对象在列表中的位置引用对象时，它被称为序号提及。这里有一个例子:

```
User: What restaurants in Berlin do you know?
Bot: Found the following objects of type ‘restaurant’: 1: I due forni 2: PastaBar 3: Berlin Burrito Company
User: Does the first one have outside seating?  
```

用户提到“I due forni”时使用的术语是“first one”。其他依次提到的词可能包括 “the second one,” “the last one,” “any,” or “3”。  
序数mentions通常用于对象列表呈现给用户时。为了将这些提及解析为实际对象，我们使用了一个序号提及映射，它被设置在KnowledgeBase类中。默认映射如下:

```
{
    "1": lambda l: l[0],
    "2": lambda l: l[1],
    "3": lambda l: l[2],
    "4": lambda l: l[3],
    "5": lambda l: l[4],
    "6": lambda l: l[5],
    "7": lambda l: l[6],
    "8": lambda l: l[7],
    "9": lambda l: l[8],
    "10": lambda l: l[9],
    "ANY": lambda l: random.choice(list),
    "LAST": lambda l: l[-1],
}
```
序数mentions映射将一个字符串(如“1”)映射到列表中的对象，例如lambda l: l[0]，表示对象在索引0处。  
例如，由于序数mentions映射不包含“the first one”项，因此使用实体同义词将NLU数据中的“the first one”映射到“1”是很重要的:

```
Does the [first one](mention:1) have [outside seating](attribute:outside-seating)?
```
NER检测到“first one”作为mention实体，但将“1”放入mention槽位。因此，我们的操作可以将提取槽位与序号提映射一起使用，以将“first one”解析为实际对象“I due forni”。  
您可以通过在您的知识库实现上调用set_ordinal_mention_mapping()函数来覆盖序号提及映射。
##### 其他Mentions
请看下面的对话:

```
User: What is the cuisine of PastaBar?
Bot: PastaBar has an Italian cuisine.
User: Does it have wifi?
Bot: Yes.
User: Can you give me an address?
```
在问题“Does it have wifi?”中，用户通过单词“it”来指代“PastaBar”。如果NER检测到实体提到的“it”，知识库操作将解析它到对话中最后提到的对象“PastaBar”。  
在下一个输入中，用户间接地引用对象“PastaBar”，而不是显式地提到它。知识库操作将检测到用户希望获取特定属性的值，在本例中是地址。如果NER没有检测到提及或对象，则该操作假定用户正在引用最近提到的对象“PastaBar”。  
您可以通过在初始化操作时将use_last_object_mention设置为False来禁用此行为。
### Customization
#### 定制ActionQueryKnowledgeBase
如果你想定制机器人对用户说的话，你可以覆盖ActionQueryKnowledgeBase的以下两个功能:
- utter_objects()
- utter_attribute_value()  

当用户请求对象列表时，使用utter_objects()。一旦机器人从知识库中检索到对象，它将在默认情况下用一条消息响应用户，格式如下:

```
Found the following objects of type ‘restaurant’: 1: I due forni 2: PastaBar 3: Berlin Burrito Company
```
如果没有找到目标，

```
I could not find any objects of type ‘restaurant’.
```
如果希望更改话语格式，可以在操作中覆盖utter_objects()方法。
函数utter_attribute_value()确定当用户请求关于对象的特定信息时，机器人发出什么话语。
如果在知识库中发现兴趣的属性，机器人将会发出以下话语:

```
‘Berlin Burrito Company’ has the value ‘Mexican’ for attribute ‘cuisine’.
```
如果未找到所请求属性的值，则bot将使用

```
Did not find a valid value for attribute ‘cuisine’ for object ‘Berlin Burrito Company’.
```
如果希望更改bot语句，可以覆盖utter_attribute_value()方法。  
**提示**：在我们的博客上有一个关于如何在自定义操作中使用知识库的[教程](https://blog.rasa.com/integrating-rasa-with-knowledge-bases/?_ga=2.141819186.1039339409.1571564377-307585668.1569235451)，该教程详细解释了ActionQueryKnowledgeBase背后的实现。  
#### 创建您自己的知识库操作
ActionQueryKnowledgeBase应该能让你轻松地开始将知识库整合到你的动作中去。但是，该动作只能处理两种用户请求:
- 用户希望从知识库中获取对象列表
- 用户希望获取特定对象的属性值

该操作不能比较对象或考虑知识库中对象之间的关系。此外，解析对话中提到的最后一个对象并不总是最优的。  
如果您想处理更复杂的用例，您可以编写自己的自定义操作。我们在rasa_sdk.knowledge_base中添加了一些辅助函数。utils([代码](https://github.com/RasaHQ/rasa-sdk/tree/master/rasa_sdk/knowledge_base/))帮助您实现自己的解决方案。我们建议使用KnowledgeBase接口，这样你就可以在自定义动作的同时使用ActionQueryKnowledgeBase了。  
如果你写了一个知识库动作，处理了上面的一个用例或一个新的用例，一定要在论坛上告诉我们!  
#### 定制InMemoryKnowledgeBase
InMemoryKnowledgeBase类继承了知识库。你可以自定义你的InMemoryKnowledgeBase覆盖以下功能:
- get_key_attribute_of_object():为了跟踪用户最后谈论的对象，我们将key属性的值存储在特定的槽中。每个对象都应该有一个惟一的键属性，类似于关系数据库中的主键。默认情况下，每种对象类型的键属性名都设置为id。您可以通过调用set_key_attribute_of_object()来覆盖特定对象类型的键属性名。
- get_representation_function_of_object():让我们关注以下餐馆:
```
{
    "id": 0,
    "name": "Donath",
    "cuisine": "Italian",
    "outside-seating": true,
    "price-range": "mid-range"
}
```
当用户要求机器人列出任何一家意大利餐馆时，它并不需要餐馆的所有细节。相反，您需要提供一个有意义的名称来标识餐馆,在大多数情况下，使用对象的名称就可以了。函数get_representation_function_of_object()返回一个lambda函数，该函数将上面的restaurant对象映射到它的名称。  

```
lambda obj: obj["name"]
```
每当机器人谈论特定对象时，都会使用此函数，以便为用户提供该对象的有意义的名称。  
默认情况下，lambda函数返回对象的“name”属性的值。如果对象没有“name”属性，或者对象的“name”不明确，那么应该通过调用set_representation_function_of_object()为该对象类型设置一个新的lambda函数。  
- set_ordinal_mention_mapping():需要序数mention映射将序数mention(例如“second one”)解析为列表中的对象。默认情况下，序数mention映射是这样的:

```
{
    "1": lambda l: l[0],
    "2": lambda l: l[1],
    "3": lambda l: l[2],
    "4": lambda l: l[3],
    "5": lambda l: l[4],
    "6": lambda l: l[5],
    "7": lambda l: l[6],
    "8": lambda l: l[7],
    "9": lambda l: l[8],
    "10": lambda l: l[9],
    "ANY": lambda l: random.choice(list),
    "LAST": lambda l: l[-1],
}
```
您可以通过调用函数set_ordinal_mention_mapping()来覆盖它。  
有关使用set_representation_function_of_object()方法覆盖对象类型“hotel”的默认表示的InMemoryKnowledgeBase的示例实现，请参见[示例bot](https://github.com/RasaHQ/rasa/blob/master/examples/knowledgebasebot/actions.py)。InMemoryKnowledgeBase的实现可以在[rasa-sdk包](https://github.com/RasaHQ/rasa-sdk/tree/master/rasa_sdk/knowledge_base/)中找到。
#### 创建自己的知识库
如果您有更多的数据，或者您想使用更复杂的数据结构，例如，涉及到不同对象之间的关系，那么您可以创建自己的知识库实现。只需继承知识库并实现get_objects()、get_object()和get_attributes_of_object()方法。[知识库代码](https://github.com/RasaHQ/rasa-sdk/tree/master/rasa_sdk/knowledge_base/)提供了关于这些方法应该做什么的更多信息。  
您还可以通过调整自定义InMemoryKnowledgeBase一节中提到的方法来进一步定制您的知识库。  
**提示**：我们写了一篇[博客](https://blog.rasa.com/set-up-a-knowledge-base-to-encode-domain-knowledge-for-rasa/?_ga=2.39443875.1039339409.1571564377-307585668.1569235451)来解释如何建立自己的知识库。

