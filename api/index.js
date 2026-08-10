// index.js - Multi-Provider OpenAI Proxy (NVIDIA NIM, DeepSeek Official, Mistral AI & OpenRouter)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();

app.use(cors());
// Set incoming payload size limit to 4.5 MB to support up to 1M token contexts
app.use(express.json({ limit: '4.5mb' }));
app.use(express.urlencoded({ limit: '4.5mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;

const MODEL_MAPPING = {
  // 1. Official DeepSeek API Direct Models
  'deepseek-v4-flash': 'deepseek-v4-flash',
  'deepseek-v4-pro': 'deepseek-v4-pro',
  'deepseek-chat': 'deepseek-chat',

  // 2. NVIDIA NIM Models
  'claude-3-opus': 'z-ai/glm-5.2', // Main Default
  'nemotron-3-ultra': 'nvidia/nemotron-3-ultra-550b-a55b',
  'laguna-xs-2.1': 'poolside/laguna-xs-2.1',
  'minimax-m3': 'minimaxai/minimax-m3',
  'gemini-pro': 'google/diffusiongemma-26b-a4b-it',
  'gpt-4-turbo': 'google/gemma-4-31b-it',
  'gpt-3.5-turbo': 'thinkingmachines/inkling',
  'mistral-medium-3.5': 'mistralai/mistral-medium-3.5-128b',

  // 3. OpenRouter Free Gemma Models
  'gemma-4-31b': 'google/gemma-4-31b-it:free',
  'gemma-4-26b': 'google/gemma-4-26b-a4b-it:free',
  'gemini-3-flash': 'google/gemma-4-31b-it:free',

  // 4. Mistral AI Official API Models
  'mistral-large-2512': 'mistral-large-latest',
  'mistral-medium-2508': 'mistral-medium-3-5'
};

function getModelConfig(nimModel, enableThinking) {
  let maxTokens = 16384; 
  let chatTemplateKwargs = undefined;
  let extraParams = {};

  const modelLower = nimModel.toLowerCase();

  if (modelLower.includes('nemotron-3-ultra')) {
    maxTokens = 16384;
    extraParams = { top_p: 0.95 };
    if (enableThinking) {
      chatTemplateKwargs = { enable_thinking: true };
      extraParams.reasoning_budget = 16384;
    } else {
      chatTemplateKwargs = { enable_thinking: false };
    }
  }
  else if (modelLower.includes('laguna-xs')) {
    maxTokens = 8192;
    extraParams = { top_p: 0.95 };
  }
  else if (modelLower.includes('minimax-m3') || modelLower.includes('minimax-3')) {
    maxTokens = 8192;
    chatTemplateKwargs = enableThinking ? { thinking_mode: "enabled" } : { thinking_mode: "disabled" };
  }
  else if (modelLower.includes('gemma-4')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { enable_thinking: true } : { enable_thinking: false };
  } 
  else if (modelLower.includes('glm-5.2')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { thinking: { type: "enabled" }, reasoning_effort: "high" } : { thinking: { type: "disabled" } };
  }
  else if (modelLower.includes('diffusiongemma') || modelLower.includes('gemma-26b')) {
    maxTokens = 4096;
    chatTemplateKwargs = enableThinking ? { enable_thinking: true } : { enable_thinking: false };
  }
  else if (modelLower.includes('inkling')) {
    maxTokens = 8192;
    chatTemplateKwargs = undefined; 
  }

  return { maxTokens, chatTemplateKwargs, extraParams };
}

app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI Proxy (NVIDIA NIM, DeepSeek, Mistral & OpenRouter)',
    default_model: 'z-ai/glm-5.2',
    dynamic_thinking: true
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'multi-provider-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream, forbiddenWords, frequency_penalty, enableThinking } = req.body;
    
    // Read enableThinking from Client UI (Default: true)
    const isThinkingEnabled = enableThinking !== undefined ? Boolean(enableThinking) : true;

    const authHeader = req.headers.authorization;
    const clientApiKey = authHeader ? authHeader.replace('Bearer ', '').trim() : '';

    let targetModel = MODEL_MAPPING[model] || model;

    // 1. Get raw messages
    let rawMessages = JSON.parse(JSON.stringify(messages));

    // 2. CONTEXT OPTIMIZATION: Clean past <think>...</think> blocks from previous assistant history
    let processedMessages = rawMessages.map(m => {
      if (m.role === 'assistant' && typeof m.content === 'string') {
        const cleanContent = m.content.replace(/<think>([\s\S]*?)(?:<\/think>|$)/gi, '').trim();
        return { role: m.role, content: cleanContent };
      }
      return m;
    });

    // 3. Construct dynamic Negative Constraint Directive if forbiddenWords provided
    let negativeConstraint = "";
    if (forbiddenWords && typeof forbiddenWords === 'string' && forbiddenWords.trim().length > 0) {
      negativeConstraint = `\n\n[CRITICAL NEGATIVE CONSTRAINTS: You are strictly forbidden from outputting or using any of the following words, phrases, AI clichés, or behaviors: ${forbiddenWords.trim()}. Choose natural, creative alternatives and adhere strictly to these exclusions.]`;
    }

    // 4. Prompt-Forced Thinking for Mistral Large (ONLY WHEN THINKING IS ENABLED)
    let forcedThinkingDirective = "";
    if (targetModel === 'mistral-large-latest' && isThinkingEnabled) {
      forcedThinkingDirective = `\n\n[REASONING DIRECTIVE: Before providing your final answer, write out your detailed step-by-step thinking process inside <think>...</think> tags.]`;
    }

    // Combine system prompt directives
    const extraSystemInstructions = negativeConstraint + forcedThinkingDirective;
    if (extraSystemInstructions.trim()) {
      const sysMsgIndex = processedMessages.findIndex(m => m.role === 'system');
      if (sysMsgIndex !== -1) {
        processedMessages[sysMsgIndex].content += extraSystemInstructions;
      } else {
        processedMessages.unshift({ role: 'system', content: extraSystemInstructions.trim() });
      }
    }

    // Determine Provider Routing
    const isOfficialDeepSeek = targetModel === 'deepseek-v4-flash' || targetModel === 'deepseek-v4-pro' || targetModel === 'deepseek-chat';
    const isMistralAPI = targetModel.startsWith('mistral-large') || targetModel.startsWith('mistral-medium-3-5');
    const isOpenRouterAPI = targetModel.includes(':free') || targetModel.startsWith('google/') && targetModel.includes(':free');

    let requestUrl = `${NIM_API_BASE}/chat/completions`;
    let requestHeaders = {
      'Authorization': `Bearer ${NIM_API_KEY || clientApiKey}`,
      'Content-Type': 'application/json'
    };
    let requestPayload = {};

    // --- 1. Official DeepSeek API Route ---
    if (isOfficialDeepSeek) {
      const activeDeepSeekKey = DEEPSEEK_API_KEY || clientApiKey;
      if (!activeDeepSeekKey) {
        return res.status(401).json({
          error: { message: 'DEEPSEEK_API_KEY is missing in Vercel environment variables.', type: 'invalid_request_error', code: 401 }
        });
      }
      requestUrl = 'https://api.deepseek.com/chat/completions';
      requestHeaders = {
        'Authorization': `Bearer ${activeDeepSeekKey}`,
        'Content-Type': 'application/json'
      };

      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: temperature !== undefined ? temperature : 0.7,
        max_tokens: max_tokens || 8192,
        stream: stream || false
      };
    }
    // --- 2. Mistral AI Official API Route ---
    else if (isMistralAPI) {
      if (!MISTRAL_API_KEY) {
        return res.status(401).json({
          error: { message: 'MISTRAL_API_KEY is missing in Vercel environment variables.', type: 'invalid_request_error', code: 401 }
        });
      }
      requestUrl = 'https://api.mistral.ai/v1/chat/completions';
      requestHeaders['Authorization'] = `Bearer ${MISTRAL_API_KEY}`;

      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: temperature !== undefined ? temperature : 0.7,
        max_tokens: max_tokens || 16384,
        stream: stream || false
      };

      if (targetModel.includes('medium') && isThinkingEnabled) {
        requestPayload.reasoning_effort = "high";
      } else {
        if (frequency_penalty !== undefined) {
          requestPayload.frequency_penalty = frequency_penalty;
        }
      }
    }
    // --- 3. OpenRouter API Route ---
    else if (isOpenRouterAPI) {
      if (!OPENROUTER_API_KEY) {
        return res.status(401).json({
          error: { message: 'OPENROUTER_API_KEY is missing in Vercel environment variables.', type: 'invalid_request_error', code: 401 }
        });
      }
      requestUrl = 'https://openrouter.ai/api/v1/chat/completions';
      requestHeaders = {
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'HTTP-Referer': 'https://tes-tis-tes.vercel.app',
        'X-Title': 'Private Chatbot Client',
        'Content-Type': 'application/json'
      };

      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: temperature !== undefined ? temperature : 0.7,
        max_tokens: max_tokens || 16384,
        stream: stream || false
      };

      if (isThinkingEnabled) {
        requestPayload.reasoning = { effort: "high" };
        requestPayload.include_reasoning = true;
      }
    }
    // --- 4. NVIDIA NIM API Route ---
    else {
      const activeApiKey = NIM_API_KEY || clientApiKey;
      if (!activeApiKey || activeApiKey === 'dummy-key') {
        return res.status(401).json({
          error: { message: 'NVIDIA API Key is missing.', type: 'invalid_request_error', code: 401 }
        });
      }

      const config = getModelConfig(targetModel, isThinkingEnabled);
      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: temperature !== undefined ? temperature : 0.7,
        max_tokens: max_tokens ? Math.min(max_tokens, config.maxTokens) : config.maxTokens,
        stream: stream || false,
        ...config.extraParams
      };

      if (config.chatTemplateKwargs) {
        requestPayload.chat_template_kwargs = config.chatTemplateKwargs;
      }
    }

    // Execute API Request
    const response = await axios.post(requestUrl, requestPayload, {
      headers: requestHeaders,
      responseType: stream ? 'stream' : 'json'
    });
    
    // --- Unified Multi-Provider Streaming Handler ---
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              const deltaChoice = data.choices?.[0]?.delta;

              if (deltaChoice) {
                let reasoningText = "";
                let contentText = "";

                if (deltaChoice.reasoning_content) reasoningText += deltaChoice.reasoning_content;
                if (deltaChoice.reasoning) reasoningText += deltaChoice.reasoning;

                if (Array.isArray(deltaChoice.content)) {
                  deltaChoice.content.forEach(item => {
                    if (item.type === 'thinking' && item.thinking) {
                      item.thinking.forEach(t => { if (t.text) reasoningText += t.text; });
                    } else if (item.type === 'text' && item.text) {
                      contentText += item.text;
                    }
                  });
                } else if (typeof deltaChoice.content === 'string') {
                  contentText += deltaChoice.content;
                }

                // Format & wrap into <think> ... </think> tags ONLY IF THINKING IS ENABLED
                if (isThinkingEnabled) {
                  let combined = '';
                  
                  if (reasoningText && !reasoningStarted) {
                    combined = '<think>\n' + reasoningText;
                    reasoningStarted = true;
                  } else if (reasoningText) {
                    combined = reasoningText;
                  }
                  
                  if (contentText && reasoningStarted) {
                    combined += '\n</think>\n\n' + contentText;
                    reasoningStarted = false;
                  } else if (contentText) {
                    combined += contentText;
                  }
                  
                  if (combined) {
                    data.choices[0].delta.content = combined;
                    delete data.choices[0].delta.reasoning_content;
                    delete data.choices[0].delta.reasoning;
                  }
                } else {
                  // If thinking is disabled, pass content only
                  data.choices[0].delta.content = contentText;
                  delete data.choices[0].delta.reasoning_content;
                  delete data.choices[0].delta.reasoning;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => res.end());
    } else {
      // Non-streaming response parsing
      let fullContent = '';
      let reasoningText = '';

      const choice = response.data.choices[0];
      if (choice) {
        if (Array.isArray(choice.message?.content)) {
          choice.message.content.forEach(item => {
            if (item.type === 'thinking' && item.thinking) {
              item.thinking.forEach(t => { if (t.text) reasoningText += t.text; });
            } else if (item.type === 'text' && item.text) {
              fullContent += item.text;
            }
          });
        } else if (typeof choice.message?.content === 'string') {
          fullContent = choice.message.content;
        }

        if (choice.message?.reasoning_content) reasoningText += choice.message.reasoning_content;
        if (choice.message?.reasoning) reasoningText += choice.message.reasoning;
      }

      if (isThinkingEnabled && reasoningText.trim()) {
        fullContent = `<think>\n${reasoningText.trim()}\n</think>\n\n` + fullContent;
      }

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: fullContent },
          finish_reason: 'stop'
        }],
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.error?.message || error.response?.data?.message || error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

module.exports = app;
