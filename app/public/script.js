(() => {
    'use strict';

    // cостояние 
    let chats = [];
    let currentChatId = null;
    let isLoading = false;
    let searchQuery = '';

    //  DOM 
    const $ = id => document.getElementById(id);
    const sidebar = $('sidebar');
    const chatList = $('chatList');
    const chatSearch = $('chatSearch');
    const chatTitle = $('chatTitle');
    const messagesEl = $('messages');
    const messageInput = $('messageInput');
    const sendBtn = $('sendBtn');
    const inputArea = $('inputArea');
    const welcomeEl = $('welcome');
    const promptModal = $('promptModal');
    const promptInput = $('promptInput');

    //  API 
    async function api(url, options = {}) {
        const res = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        return data;
    }

    //  kогирование на бэк
    function log(level, message, meta = {}) {
        fetch('/api/log', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ level, message, meta })
        }).catch(() => {});
    }

    // рендер списока чатов 
    function renderChats() {
        const filtered = searchQuery
            ? chats.filter(c => c.name.toLowerCase().includes(searchQuery.toLowerCase()))
            : chats;

        chatList.innerHTML = '';

        if (!filtered.length) {
            chatList.innerHTML = '<div style="padding:16px;color:var(--fg2);font-size:13px;text-align:center">Нет чатов</div>';
            return;
        }

        filtered.forEach(chat => {
            const item = document.createElement('div');
            item.className = 'chat-item' + (chat.id === currentChatId ? ' active' : '');
            item.innerHTML = `
                <span class="chat-item-name"></span>
                <button class="chat-item-del" title="Удалить">✕</button>
            `;
            item.querySelector('.chat-item-name').textContent = chat.name;
            item.querySelector('.chat-item-name').onclick = () => selectChat(chat.id);
            item.querySelector('.chat-item-del').onclick = e => {
                e.stopPropagation();
                deleteChat(chat.id);
            };
            chatList.appendChild(item);
        });
    }

    // рендер сообщений
    function renderMessages(messages) {
        messagesEl.innerHTML = '';
        messages.forEach(msg => appendMessage(msg.role, msg.content));
        scrollToBottom();
    }

    function appendMessage(role, content) {
        if (welcomeEl && welcomeEl.parentNode) welcomeEl.remove();

        const el = document.createElement('div');
        el.className = `message ${role}`;
        el.textContent = content;
        messagesEl.appendChild(el);
        scrollToBottom();
        return el;
    }

    function scrollToBottom() {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function showError(text) {
        const el = document.createElement('div');
        el.className = 'message error';
        el.textContent = text;
        messagesEl.appendChild(el);
        scrollToBottom();
    }

    // загрузка чатов 
    async function loadChats() {
        try {
            const data = await api('/api/chats');
            chats = data.chats || [];
            renderChats();
        } catch (e) {
            log('error', 'Failed to load chats', { error: e.message });
        }
    }

    // выбор чата 
    async function selectChat(chatId) {
        if (isLoading) return;
        currentChatId = chatId;

        const chat = chats.find(c => c.id === chatId);
        chatTitle.textContent = chat ? chat.name : 'Чат';

        renderChats();
        messagesEl.innerHTML = '<div style="color:var(--fg2);text-align:center;padding:20px">Загрузка...</div>';

        try {
            const messages = await api(`/api/chat/${chatId}`);
            renderMessages(messages);
            inputArea.classList.remove('hidden');
            messageInput.focus();
        } catch (e) {
            messagesEl.innerHTML = '';
            showError('Не удалось загрузить историю');
            log('error', 'Failed to load messages', { chatId, error: e.message });
        }

        if (window.innerWidth <= 768) sidebar.classList.remove('open');
    }

    // создание чата
    async function createChat() {
        if (isLoading) return;
        try {
            const data = await api('/api/chat/new', {
                method: 'POST',
                body: JSON.stringify({})
            });
            chats.unshift({ id: data.id, name: data.name });
            renderChats();
            await selectChat(data.id);
            log('info', 'Chat created', { chatId: data.id });
        } catch (e) {
            showError('Не удалось создать чат');
            log('error', 'Failed to create chat', { error: e.message });
        }
    }

    //удаление чата
    async function deleteChat(chatId) {
        if (!confirm('Удалить этот чат?')) return;
        try {
            await api(`/api/chat/${chatId}`, { method: 'DELETE' });
            chats = chats.filter(c => c.id !== chatId);
            if (currentChatId === chatId) {
                currentChatId = null;
                chatTitle.textContent = 'Выберите чат';
                messagesEl.innerHTML = '<div id="welcome">Выберите чат или создайте новый</div>';
                inputArea.classList.add('hidden');
            }
            renderChats();
            log('info', 'Chat deleted', { chatId });
        } catch (e) {
            showError('Не удалось удалить чат');
            log('error', 'Failed to delete chat', { chatId, error: e.message });
        }
    }

    //отправка сообщения 
    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || !currentChatId || isLoading) return;

        // Команда exit
        if (text.toLowerCase() === 'exit') {
            try { await api(`/api/chat/${currentChatId}/message`, {
                method: 'POST',
                body: JSON.stringify({ message: text })
            }); } catch {}
            alert('Сервер останавливается...');
            return;
        }

        isLoading = true;
        sendBtn.disabled = true;
        messageInput.value = '';
        messageInput.style.height = 'auto';

        appendMessage('user', text);

        // индикатор того что ии печатает ответ
        const typing = appendMessage('assistant', '…');

        try {
            const data = await api(`/api/chat/${currentChatId}/message`, {
                method: 'POST',
                body: JSON.stringify({ message: text })
            });
            typing.textContent = data.content;
            log('info', 'Message sent', { chatId: currentChatId });
        } catch (e) {
            typing.remove();
            showError('Ошибка: ' + e.message);
            log('error', 'Failed to send message', { chatId: currentChatId, error: e.message });
        } finally {
            isLoading = false;
            sendBtn.disabled = messageInput.value.trim() === '';
            messageInput.focus();
        }
    }

    //системный промпт 
    function openPromptModal() {
        const chat = chats.find(c => c.id === currentChatId);
        if (!chat) return;
        promptInput.value = chat.systemPrompt || '';
        promptModal.classList.add('open');
    }

    async function savePrompt() {
        try {
            await api(`/api/chat/${currentChatId}/system-prompt`, {
                method: 'PUT',
                body: JSON.stringify({ systemPrompt: promptInput.value })
            });
            const chat = chats.find(c => c.id === currentChatId);
            if (chat) chat.systemPrompt = promptInput.value;
            promptModal.classList.remove('open');
            log('info', 'System prompt saved', { chatId: currentChatId });
        } catch (e) {
            alert('Не удалось сохранить промпт');
            log('error', 'Failed to save prompt', { error: e.message });
        }
    }

    //обработчики 
    $('newChatBtn').onclick = createChat;
    $('deleteChatBtn').onclick = () => currentChatId && deleteChat(currentChatId);
    $('systemPromptBtn').onclick = openPromptModal;
    $('promptCancel').onclick = () => promptModal.classList.remove('open');
    $('promptSave').onclick = savePrompt;
    $('toggleSidebar').onclick = () => sidebar.classList.toggle('open');

    chatSearch.oninput = e => {
        searchQuery = e.target.value;
        renderChats();
    };

    messageInput.oninput = () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
        sendBtn.disabled = messageInput.value.trim() === '' || isLoading;
    };

    messageInput.onkeydown = e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    sendBtn.onclick = sendMessage;

    promptModal.onclick = e => {
        if (e.target === promptModal) promptModal.classList.remove('open');
    };

    // старт 
    inputArea.classList.add('hidden');
    loadChats();
    log('info', 'Frontend initialized');
})();