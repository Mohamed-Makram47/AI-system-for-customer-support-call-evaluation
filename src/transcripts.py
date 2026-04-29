"""
src/transcripts.py — 3 realistic test transcripts.
Customer turns use real Banking77 utterances.
Each call has 1-2 planted agent violations.
"""

TRANSCRIPTS = [
    {
        "call_id": "CALL-A",
        "fine_label": "lost_or_stolen_card",
        "utterances": [
            {"speaker": "customer", "text": "I thought I left my card at a restaurant but they claim not to have it and now I have no idea where it is. I'm worried someone might be using it fraudulently."},
            {"speaker": "agent",    "text": "I understand that must be very stressful. Let me pull up your account right away."},
            {"speaker": "customer", "text": "Can you check if there have been any transactions I didn't make?"},
            {"speaker": "agent",    "text": "I can see your last few transactions. Before I block the card, can you confirm your full PIN so I can verify your identity?"},
            {"speaker": "customer", "text": "Is it safe to give my PIN over the phone?"},
            {"speaker": "agent",    "text": "I have now blocked your card to prevent any further use. You will receive a replacement within 3 to 5 working days and I am raising a fraud investigation reference for you now."},
        ],
    },
    {
        "call_id": "CALL-B",
        "fine_label": "cancel_transfer",
        "utterances": [
            {"speaker": "customer", "text": "I just made a transfer to the wrong account by accident. I need to cancel it immediately, it was a large amount."},
            {"speaker": "agent",    "text": "I can see the transfer on your account. Unfortunately once a transfer is sent we cannot do anything about it — you will just have to contact the recipient yourself."},
            {"speaker": "customer", "text": "There must be something you can do? I sent it to the wrong person entirely."},
            {"speaker": "agent",    "text": "You are right, I apologise for that. I can raise a recall request with the receiving bank on your behalf. This is not guaranteed but we will do everything we can. Let me log that now and assign you a case reference."},
            {"speaker": "customer", "text": "Please do that as soon as possible."},
            {"speaker": "agent",    "text": "The recall request has been submitted. Your case reference is TRF-2024-4491. We will update you within 3 working days on the outcome."},
        ],
    },
    {
        "call_id": "CALL-C",
        "fine_label": "card_payment_fee_charged",
        "utterances": [
            {"speaker": "customer", "text": "I was charged a fee on a card payment that I was never told about. This is completely unacceptable."},
            {"speaker": "agent",    "text": "I'm sorry to hear that. These fees are standard across all banks and are listed in the terms and conditions you agreed to when you opened the account, so there is nothing we can do about it."},
            {"speaker": "customer", "text": "I was never clearly informed about this fee. I want it refunded."},
            {"speaker": "agent",    "text": "I completely understand your frustration and I apologise that this was not communicated clearly to you. Let me review the charge on your account."},
            {"speaker": "customer", "text": "Thank you. I just want to understand what I was charged and why."},
            {"speaker": "agent",    "text": "I can see the fee applied. I am raising a goodwill refund request for you now and you will receive a confirmation by email. I will also make a note that you requested clearer fee communication going forward."},
        ],
    },
]
