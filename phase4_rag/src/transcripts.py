"""
src/transcripts.py — Fake call transcripts for RAG evaluation testing.
Each transcript has one clear agent violation and one good response.
"""

TRANSCRIPTS = [
    {
        "call_id": "CALL-001",
        "fine_label": "lost_or_stolen_card",
        "utterances": [
            {"speaker": "customer", "text": "Hi, I think my card has been stolen. I noticed some charges I didn't make."},
            {"speaker": "agent",    "text": "I'll go ahead and block your card right now to stop any further transactions."},
            {"speaker": "customer", "text": "Thank you. What do I do about those unauthorized charges?"},
            {"speaker": "agent",    "text": "Don't worry, you're fully liable for all charges on your account until you formally report the theft in writing."},
            {"speaker": "customer", "text": "Really? I thought I was protected."},
            {"speaker": "agent",    "text": "Let me also arrange a replacement card for you. You should receive it within 5 to 7 business days and I'll give you a reference number now."},
        ],
    },
    {
        "call_id": "CALL-002",
        "fine_label": "change_pin",
        "utterances": [
            {"speaker": "customer", "text": "I'd like to change my PIN. I think someone may have seen it."},
            {"speaker": "agent",    "text": "Of course. Can you confirm your date of birth and the last four digits of your account number so I can verify your identity?"},
            {"speaker": "customer", "text": "Sure, it's March 14th 1990 and the last four are 7823."},
            {"speaker": "agent",    "text": "Great, I've verified you. Your new PIN will be 4521 — please use that going forward."},
            {"speaker": "customer", "text": "Wait, you're just telling me my new PIN over the phone?"},
            {"speaker": "agent",    "text": "You're right, I apologize. For security, PIN changes must be done through the app or an ATM. I should not have disclosed any PIN information over the phone."},
        ],
    },
    {
        "call_id": "CALL-003",
        "fine_label": "declined_card_payment",
        "utterances": [
            {"speaker": "customer", "text": "My card was declined at a restaurant just now. It was really embarrassing."},
            {"speaker": "agent",    "text": "I'm sorry to hear that. Let me look into your account right away."},
            {"speaker": "customer", "text": "Do you know why it happened?"},
            {"speaker": "agent",    "text": "It looks like the transaction was flagged by our fraud system. To be honest, your spending pattern looked suspicious so the system just blocks cards like yours automatically."},
            {"speaker": "customer", "text": "That's not very reassuring. Can you unblock it?"},
            {"speaker": "agent",    "text": "Absolutely. I've cleared the flag and your card is now active again. I'd also recommend enabling transaction notifications in the app so you're alerted immediately if this happens again."},
        ],
    },
    {
        "call_id": "CALL-004",
        "fine_label": "request_refund",
        "utterances": [
            {"speaker": "customer", "text": "I was charged twice for the same purchase yesterday. I need a refund."},
            {"speaker": "agent",    "text": "I can see both charges on your account. I'll raise a dispute for the duplicate transaction right away."},
            {"speaker": "customer", "text": "How long will the refund take?"},
            {"speaker": "agent",    "text": "It usually takes 3 to 5 business days, but honestly there's no guarantee — these things can take weeks and sometimes we just can't get the money back."},
            {"speaker": "customer", "text": "That's worrying. Is there a reference number I can track?"},
            {"speaker": "agent",    "text": "Yes, your dispute reference is DIS-2024-8841. You can use that to follow up with us at any time and we will keep you updated on the outcome."},
        ],
    },
    {
        "call_id": "CALL-005",
        "fine_label": "cancel_transfer",
        "utterances": [
            {"speaker": "customer", "text": "I just sent a transfer to the wrong account by mistake. Can you cancel it?"},
            {"speaker": "agent",    "text": "I understand, let me check the status of that transfer immediately."},
            {"speaker": "customer", "text": "Please hurry, it was for 500 pounds."},
            {"speaker": "agent",    "text": "The transfer has already been processed and left our system. There's nothing we can do — once it's gone it's gone, and we take no responsibility for misdirected transfers."},
            {"speaker": "customer", "text": "There must be something you can do?"},
            {"speaker": "agent",    "text": "You're right, I apologize for that. We can raise a recall request with the receiving bank on your behalf. I'll log that now and you'll receive a case number within 24 hours with next steps."},
        ],
    },
]
