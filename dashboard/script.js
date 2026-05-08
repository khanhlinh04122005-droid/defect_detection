document.addEventListener("DOMContentLoaded", () => {
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size   = 11;
    Chart.defaults.color       = "#64748b";

    // ── Fabric Defect Data ──
    const FABRIC_DEFECT_TYPES = ['Thủng lỗ', 'Xổ chỉ', 'Loang màu', 'Lỗi dệt', 'Xơ vải'];
    const DEFECT_COLORS = ['#ef4444', '#8b5cf6', '#f97316', '#3b82f6', '#94a3b8'];

    // 1. Pie/Donut Chart — Tỷ lệ loại lỗi vải
    const pieCtx = document.getElementById('defectPieChart').getContext('2d');
    new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: FABRIC_DEFECT_TYPES,
            datasets: [{
                data: [35, 25, 20, 12, 8],
                backgroundColor: DEFECT_COLORS,
                borderWidth: 2,
                borderColor: '#fff',
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${ctx.label}: ${ctx.raw}%`
                    }
                }
            }
        }
    });

    // 2. Bar Chart — Số lượng lỗi theo loại vải
    const barCtx = document.getElementById('defectBarChart').getContext('2d');
    new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: ['Cotton', 'Denim', 'Lụa', 'Len', 'Thun', 'Tổng hợp'],
            datasets: [{
                label: 'Số lỗi',
                data: [24, 18, 8, 12, 15, 10],
                backgroundColor: [
                    '#6366f1', '#8b5cf6', '#ec4899',
                    '#f59e0b', '#10b981', '#3b82f6'
                ],
                borderRadius: 5,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 9 } }
                },
                y: {
                    beginAtZero: true,
                    border: { display: false },
                    grid: { color: '#f1f5f9' },
                    ticks: { font: { size: 9 }, stepSize: 5 }
                }
            }
        }
    });

    // 3. Line Chart — Xu hướng lỗi theo thời gian (tuần)
    const lineCtx = document.getElementById('defectLineChart').getContext('2d');
    new Chart(lineCtx, {
        type: 'line',
        data: {
            labels: ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
            datasets: [
                {
                    label: 'Tổng lỗi',
                    data: [28, 35, 22, 40, 30, 18],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99,102,241,0.08)',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: true
                },
                {
                    label: 'Thủng lỗ',
                    data: [10, 15, 8, 18, 12, 6],
                    borderColor: '#ef4444',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 9 } }
                },
                y: {
                    beginAtZero: true,
                    border: { display: false },
                    grid: { color: '#f1f5f9' },
                    ticks: { font: { size: 9 } }
                }
            }
        }
    });

    // ── Image Upload ──
    const fileInput    = document.getElementById('file-upload');
    const inputImage   = document.getElementById('input-image');
    const inputPlaceholder = document.getElementById('input-placeholder');

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (ev) => {
            inputImage.src = ev.target.result;
            inputImage.style.display = 'block';
            if (inputPlaceholder) inputPlaceholder.style.display = 'none';
        };
        reader.readAsDataURL(file);
    });

    // ── Analyze Button (Demo) ──
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.addEventListener('click', () => {
        const fabricType = document.getElementById('fabric-type').value || 'cotton';
        const aiModel    = document.getElementById('ai-model').value;

        // Show analyzing state
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang phân tích...';
        analyzeBtn.disabled = true;

        setTimeout(() => {
            analyzeBtn.innerHTML = '<i class="fas fa-microscope"></i> Phân tích';
            analyzeBtn.disabled = false;

            // Update info based on selected fabric
            const fabricLabels = {
                cotton: 'Vải cotton', denim: 'Vải denim / Jean',
                silk: 'Lụa / Vải mỏng', wool: 'Len / Vải dày',
                synthetic: 'Vải tổng hợp', knit: 'Vải thun / Dệt kim',
                leather: 'Da tổng hợp', carpet: 'Thảm / Vải dày'
            };
            const infoFabric = document.getElementById('info-fabric');
            if (infoFabric) infoFabric.textContent = fabricLabels[fabricType] || 'Vải cotton';

            // Add result message to chat
            addAIMessage('Phân tích hoàn tất! Phát hiện 3 vùng lỗi trên tấm vải. Verdict: FAIL — Major defects detected.');
        }, 2000);
    });

    // ── Chatbot ──
    const chatInput  = document.getElementById('chat-input');
    const sendBtn    = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');

    const FABRIC_ANSWERS = [
        'Lỗi thủng lỗ trên vải cotton thường do kim đan sai hoặc lực căng không đồng đều trong quá trình dệt.',
        'Vải bị loang màu có thể do thuốc nhuộm không đều hoặc nhiệt độ sấy quá cao trong quá trình xử lý.',
        'Xổ chỉ thường xảy ra khi chất lượng sợi kém hoặc độ bền kéo thấp hơn tiêu chuẩn quy định.',
        'Tiêu chuẩn AATCC và ISO 139 thường được dùng để đánh giá chất lượng vải trong sản xuất may mặc.',
        'Lỗi xơ vải (pilling) xuất hiện sau quá trình ma sát, thường gặp ở vải tổng hợp kém chất lượng.',
        'Hệ thống AI hiện hỗ trợ phát hiện 9 loại lỗi vải: thủng, rách, loang màu, xổ chỉ, lỗi dệt, xơ vải, bạc màu, bẩn dầu mỡ và lỗi khác.'
    ];
    let answerIdx = 0;

    function addUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'message user-message';
        div.textContent = text;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addAIMessage(text) {
        const div = document.createElement('div');
        div.className = 'message ai-message';
        div.innerHTML = `<i class="fas fa-robot bot-icon"></i>${text}`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;
        addUserMessage(text);
        chatInput.value = '';
        setTimeout(() => {
            addAIMessage(FABRIC_ANSWERS[answerIdx % FABRIC_ANSWERS.length]);
            answerIdx++;
        }, 800);
    }

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});
