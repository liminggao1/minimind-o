document.querySelectorAll("[data-reveal]").forEach((button) => {
    button.addEventListener("click", () => {
        const answer = document.getElementById(button.dataset.reveal);
        const isVisible = answer.classList.toggle("is-visible");
        button.textContent = isVisible ? "收起答案" : "核对答案";
        button.setAttribute("aria-expanded", String(isVisible));
    });
});
