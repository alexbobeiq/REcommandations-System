document.getElementById('btnSearch').addEventListener('click', loadRecommandations);

async function loadRecommandations() {
    const customerId = document.getElementById('customerId').value;
    const grid = document.getElementById('productsGrid');
    const loading = document.getElementById('loading');

    if (!customerId) {
        alert("Please insert a valid ID");
        return;
    }

    grid.innerHTML = '';
    loading.classList.remove('hidden');

    try {
        const url = `https://recommandations-system.onrender.com/store/recommandations/${customerId}`;
        const response = await fetch(url);

        if (!response.ok) throw new Error("Server not responding");

        const produse = await response.json();
        loading.classList.add('hidden');

        if (produse.length === 0) {
            grid.innerHTML = '<p style="grid-column: 1/-1">No recommandations for this client</p>';
            return;
        }

        produse.forEach(p => {
            const card = document.createElement('div');
            card.className = 'product-card';


            card.innerHTML = `
                <div class="product-code">COD: ${p.code}</div>
                <div class="product-name">${p.name}</div>
                <div class="product-price">${p.price.toFixed(2)} $</div>
                <button onclick="alert('Added to cart: ${p.name}')">Add to cart</button>
            `;
            grid.appendChild(card);
        });

    } catch (error) {
        loading.classList.add('hidden')
    }
}

function setDemoClient(id) {
    document.getElementById('customerId').value = id;
    loadRecommandations();
}