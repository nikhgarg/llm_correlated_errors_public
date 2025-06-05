model_configs = {
#   'anthropic.claude-v2': {
#     "config": {
#       "max_tokens_to_sample": 300,
#       "temperature": 0.1,
#       "top_p": 0.9
#     },
#     "cost_mapping": (.008/1000, .024/1000),
#     "keyword": "completion",
#     "short_name": "Claude"
#   },
  'amazon.titan-text-lite-v1': {
      "config": {
          "temperature": 0.0,
          "topP": 0.9
      },
      "cost_mapping": (0.00015/1000, 0.0002/1000),
      "keyword": "outputText",
      "short_name": "Titan-Lit"
  },
  'amazon.titan-text-express-v1': {
      "config": {
          "temperature": 0.0,
          "topP": 0.9
      },
      "cost_mapping": (0.0002/1000, 0.0006/1000),
      "keyword": "outputText",
      "short_name": "Titan-Exp"
  },
  'us.amazon.nova-lite-v1:0': {
      "config": {
          "temperature": 0.0,
          "topP": 0.9
      },
      "cost_mapping": (0.00004/1000, 0.00014/1000),
      "keyword": "text",
      "short_name": "Nova-Lite"
  },
  'us.amazon.nova-micro-v1:0': {
      "config": {
          "temperature": 0.0,
          "topP": 0.9
      },
      "cost_mapping": (0.00004/1000, 0.00014/1000),
      "keyword": "text",
      "short_name": "Nova-Micro"
  },
  'us.amazon.nova-pro-v1:0': {
      "config": {
          "temperature": 0.0,
          "topP": 0.9
      },
      "cost_mapping": (0.0008/1000, 0.0032/1000),
      "keyword": "text",
      "short_name": "Nova-Pro"
  },
  'mistral.mistral-7b-instruct-v0:2': {
    "config": {
        "temperature": 0.0,
        "top_p": 0.9
    },
    "cost_mapping": (0.003/1000, 0.009/1000),
    "keyword": "text",
    "short_name": "Mistral-7B-Instruct"
  },
  'mistral.mistral-large-2402-v1:0': {
    "config": {
        "temperature": 0.0,
        "top_p": 0.9
    },
    "cost_mapping": (0.003/1000, 0.009/1000),
    "keyword": "text",
    "short_name": "Mistral-Large(24.02)"
  },
  'mistral.mistral-large-2407-v1:0': {
    "config": {
        "temperature": 0.0,
        "top_p": 0.9
    },
    "cost_mapping": (0.003/1000, 0.009/1000),
    "keyword": "content",
    "short_name": "Mistral-Large"
  },
  'mistral.mixtral-8x7b-instruct-v0:1': {
    "config": {
        "temperature": 0.0,
        "top_p": 0.9
    },
    "cost_mapping": (0.0007/1000, 0.0007/1000),
    "keyword": "text",
    "short_name": "Mistral-8x7b"
  },
  'meta.llama3-70b-instruct-v1:0': {
    "config": {
      "temperature": 0.0,
      "top_p": 0.9
    },
    "cost_mapping": (0.00265/1000, 0.0035/1000),
    "keyword": "generation",
    "short_name": "Llama3-70b"
  },
  'meta.llama3-1-8b-instruct-v1:0': {
    "config": {
      "temperature": 0.0,
      "top_p": 0.9
    },
    "cost_mapping": (0.0003/1000, 0.0006/1000),
    "keyword": "generation",
    "short_name": "Llama3-1-8b"
  },
  'meta.llama3-1-70b-instruct-v1:0': {
    "config": {
      "temperature": 0.0,
      "top_p": 0.9
    },
    "cost_mapping": (0.00265/1000, 0.0035/1000),
    "keyword": "generation",
    "short_name": "Llama3-1-70b"
  },
  'meta.llama3-1-405b-instruct-v1:0': {
    "config": {
      "temperature": 0.0,
      "top_p": 0.9
    },
    "cost_mapping": (0.00532/1000, 0.016/1000),
    "keyword": "generation",
    "short_name": "Llama3-1-405b"
  },
  'us.meta.llama3-2-11b-instruct-v1:0': {
    "config": {
      "temperature": 0.0,
      "top_p": 0.9
    },
    "cost_mapping": (0.00035/1000, 0.00035/1000),
    "keyword": "generation",
    "short_name": "Llama3-2-11b"
  },
  'us.meta.llama3-2-90b-instruct-v1:0': {
    "config": {
      "temperature": 0.0,
      "top_p": 0.9
    },
    "cost_mapping": (0.002/1000, 0.002/1000),
    "keyword": "generation",
    "short_name": "Llama3-2-90b"
  },
  'us.meta.llama3-3-70b-instruct-v1:0': {
    "config": {
      "temperature": 0.0,
      "top_p": 0.9
    },
    "cost_mapping": (0.0001/1000, 0.0004/1000),
    "keyword": "generation",
    "short_name": "Llama3-3-70b"
  },
  'ai21.jamba-1-5-large-v1:0': {
    "config": {
      "temperature": 0.1,
      "top_p": 0.9
    },
    "cost_mapping": (0.002/1000, 0.008/1000),
    "keyword": "content",
    "short_name": "Jamba1.5-Large"
  },
  'ai21.jamba-1-5-mini-v1:0': {
    "config": {
      "temperature": 0.1,
      "top_p": 0.9
    },
    "cost_mapping": (0.0002/1000, 0.0004/1000),
    "keyword": "content",
    "short_name": "Jamba1.5-Mini"
  },
}
