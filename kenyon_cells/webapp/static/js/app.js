document.addEventListener('DOMContentLoaded', () => {
    let pageStates = {}; // Object to store the state of each page

    function renderTable() {
        const tableBody = document.getElementById('tableBody');
        const tableHead = document.querySelector('#annotationTable thead tr');
        tableBody.innerHTML = '';
        tableHead.innerHTML = '<th>Item</th>';

        columns.forEach(column => {
            const th = document.createElement('th');
            th.textContent = column;
            tableHead.appendChild(th);
        });

        Object.entries(items).forEach(([item, states]) => {
            const row = document.createElement('tr');
            const imgCell = document.createElement('td');
            imgCell.innerHTML = `<img src="${images[item]}" alt="${item}"><br>${item}`;
            row.appendChild(imgCell);

            const pageState = pageStates[item] ? pageStates[item] : states;

            pageState.forEach((state, index) => {
                const cell = document.createElement('td');
                cell.innerHTML = `<input type="checkbox" name="state_${item}_${index}" ${state ? 'checked' : ''}>`;
                cell.querySelector('input').addEventListener('change', (event) => {
                    if (!pageStates[item]) {
                        pageStates[item] = Array(states.length).fill(false);
                    }
                    pageStates[item][index] = event.target.checked;
                });
                row.appendChild(cell);
            });

            tableBody.appendChild(row);
        });
    }

    document.getElementById('annotationForm').addEventListener('submit', (event) => {
        const pageStatesInput = document.getElementById('pageStatesInput');
        pageStatesInput.value = JSON.stringify(pageStates);
    });

    // Initial render
    renderTable();
});