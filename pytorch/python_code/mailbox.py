# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 23:47:07 2020

@author: aaaambition
"""

from collections import defaultdict

class Mailbox:
    def __init__(self):
        # Format of boxes is box[id] -> [message1, message2, ...]
        self.inbox = defaultdict(lambda: [])
        self.outbox = defaultdict(lambda: [])
        
    def send_message(self, message, recepient_id):
        """
        Queues up message in the outbox to be sent, note actual sending is done
        by the pytorch server
        """
        self.outbox[recepient_id].append(message)

    def receive_message(self, message, sender_id):
        """
        Adds a message to the inbox
        """
        self.inbox[sender_id].append(message)
        
    def get_inbox_messages(self):
        """
        Returns a list of messages in the format (sender, message)
        """
        msg_list = []
        for sender, messages in self.inbox.items():
            for message in messages:
                msg_list.append((sender, message))
        
        return msg_list
        
    def clear_inbox(self):
        self.inbox = defaultdict(lambda: [])
        
    def clear_outbox(self):
        self.outbox = defaultdict(lambda: [])
        
        