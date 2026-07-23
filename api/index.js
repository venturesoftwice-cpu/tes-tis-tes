// index.js - OpenAI to NVIDIA NIM, OpenRouter & Mistral AI Proxy
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

const SHOW_REASONING = true; 
const ENABLE_THINKING_MODE = true; 

const MODEL_MAPPING = {
  // NVIDIA NIM Models
  'gpt-3.5-turbo': 'thinkingmachines/inkling',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'google/gemma-4-31b-it',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro', 
  'claude-3-opus': 'z-ai/glm-5.2',
  'claude-3-sonnet': 'deepseek-ai/deepseek-v4-flash',
  'gemini-pro': 'google/diffusiongemma-26b-a4b-it',
  'mistral-medium-3.5': 'mistralai/mistral-medium-3.5-128b',

  // MiniMax NVIDIA NIM Models
  'minimax-m3': 'minimaxai/minimax-m3',
  'minimax-m2.7': 'minimaxai/minimax-m2.7',

  // OpenRouter Free Gemma Models
  'gemma-4-31b': 'google/gemma-4-31b-it:free',
  'gemma-4-26b': 'google/gemma-4-26b-a4b-it:free',
  'gemini-3-flash': 'google/gemma-4-31b-it:free',

  // Mistral AI Official API Models
  'mistral-large-2512': 'mistral-large-latest',
  'mistral-medium-2508': 'mistral-medium-3-5'
};

function getModelConfig(nimModel, enableThinking) {
  let maxTokens = 16384; 
  let chatTemplateKwargs = undefined;

  const modelLower = nimModel.toLowerCase();

  if (modelLower.includes('minimax-m3') || modelLower.includes('minimax-3')) {
    maxTokens = 8192;
    chatTemplateKwargs = enableThinking ? { thinking_mode: "enabled" } : { thinking_mode: "disabled" };
  }
  else if (modelLower.includes('minimax-m2.7') || modelLower.includes('minimax-2.7')) {
    maxTokens = 8192;
    chatTemplateKwargs = undefined;
  }
  else if (modelLower.includes('gemma-4')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { enable_thinking: true } : undefined;
  } 
  else if (modelLower.includes('deepseek-v4') || modelLower.includes('deepseek')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { thinking: true, reasoning_effort: "high" } : undefined;
  }
  else if (modelLower.includes('inkling')) {
    maxTokens = 8192;
    chatTemplateKwargs = undefined; 
  }
  else if (modelLower.includes('glm-5.2')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { thinking: { type: "enabled" }, reasoning_effort: "high" } : undefined;
  }
  else if (modelLower.includes('diffusiongemma') || modelLower.includes('gemma-26b')) {
    maxTokens = 4096;
    chatTemplateKwargs = enableThinking ? { enable_thinking: true } : undefined;
  }

  return { maxTokens, chatTemplateKwargs };
}

app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM, OpenRouter & Mistral AI Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
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
    const { model, messages, temperature, max_tokens, stream, forbiddenWords, frequency_penalty } = req.body;
    
    const authHeader = req.headers.authorization;
    const clientApiKey = authHeader ? authHeader.replace('Bearer ', '').trim() : '';

    let targetModel = MODEL_MAPPING[model] || model;

    // 1. Dapatkan pesan mentah
    let rawMessages = JSON.parse(JSON.stringify(messages));

    // 2. OPTIMASI KONTEKS: Bersihkan tag <think>...</think> dari riwayat jawaban asisten terdahulu
    let processedMessages = rawMessages.map(m => {
      if (m.role === 'assistant' && typeof m.content === 'string') {
        const cleanContent = m.content.replace(/<think>([\s\S]*?)(?:<\/think>|$)/gi, '').trim();
        return { role: m.role, content: cleanContent };
      }
      return m;
    });

    // 3. Construct dynamic Negative Constraint Directive jika forbiddenWords diisi
    let negativeConstraint = "";
    if (forbiddenWords && typeof forbiddenWords === 'string' && forbiddenWords.trim().length > 0) {
      negativeConstraint = `\n\n[CRITICAL NEGATIVE CONSTRAINTS: You are strictly forbidden from outputting or using any of the following words, phrases, AI clichés, or behaviors: ${forbiddenWords.trim()}. Choose natural, creative alternatives and adhere strictly to these exclusions.]`;
    }

    // 4. Prompt-Forced Thinking untuk Mistral Large (memaksa pemikiran analitis sebelum menjawab)
    let forcedThinkingDirective = "";
    if (targetModel === 'mistral-large-latest') {
      forcedThinkingDirective = `\n\n[REASONING DIRECTIVE: Before providing your final answer, write out your detailed step-by-step thinking process inside <think>...</think> tags.]`;
    }

    // Gabungkan instruksi tambahan ke System Prompt
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
    const isMistralAPI = targetModel.startsWith('mistral-large') || targetModel.startsWith('mistral-medium-3-5');
    const isOpenRouterAPI = targetModel.includes(':free') || targetModel.startsWith('google/');

    let requestUrl = `${NIM_API_BASE}/chat/completions`;
    let requestHeaders = {
      'Authorization': `Bearer ${NIM_API_KEY || clientApiKey}`,
      'Content-Type': 'application/json'
    };
    let requestPayload = {};

    // --- 1. Mistral AI Official API Route ---
    if (isMistralAPI) {
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

      if (targetModel.includes('medium')) {
        requestPayload.reasoning_effort = "high";
      } else {
        if (frequency_penalty !== undefined) {
          requestPayload.frequency_penalty = frequency_penalty;
        }
      }
    }
    // --- 2. OpenRouter API Route ---
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
        reasoning: { effort: "high" },
        include_reasoning: true,
        stream: stream || false
      };
    }
    // --- 3. NVIDIA NIM API Route ---
    else {
      const activeApiKey = NIM_API_KEY || clientApiKey;
      if (!activeApiKey || activeApiKey === 'dummy-key') {
        return res.status(401).json({
          error: { message: 'NVIDIA API Key is missing.', type: 'invalid_request_error', code: 401 }
        });
      }

      const config = getModelConfig(targetModel, ENABLE_THINKING_MODE);
      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: temperature !== undefined ? temperature : 0.7,
        max_tokens: max_tokens ? Math.min(max_tokens, config.maxTokens) : config.maxTokens,
        stream: stream || false
      };

      if (config.chatTemplateKwargs) {
        requestPayload.chat_template_kwargs = config.chatTemplateKwargs;
      }

      if (targetModel.toLowerCase().includes('mistral-medium-3.5')) {
        requestPayload.reasoning_effort = "high";
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

                // Parse OpenRouter & NIM reasoning parameters
                if (deltaChoice.reasoning_content) reasoningText += deltaChoice.reasoning_content;
                if (deltaChoice.reasoning) reasoningText += deltaChoice.reasoning;

                // Parse Mistral AI stream array structure
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

                // Format & wrap into <think> ... </think> tags
                if (SHOW_REASONING) {
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

      if (SHOW_REASONING && reasoningText.trim()) {
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
