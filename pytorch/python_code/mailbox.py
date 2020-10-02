# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 23:47:07 2020

@author: aaaambition
"""

from collections import defaultdict, namedtuple

class Mailbox:
    Message = namedtuple('Message', 'sender recepient contents')
    def __init__(self, contacts, dead_channel_code):
        # Format of boxes is box[id] -> [message1, message2, ...]
        # An iterable describing who may contact who. Entries in format {sender: [receivers]}
        self.contacts = contacts
        self.inbox = defaultdict(lambda: [])
        # prefill everyone's mailbox with an initial message
        for sender, recepients in self.contacts.items():
            for recepient in recepients:
                self.inbox[recepient].append(self.Message(sender, recepient, dead_channel_code))
            
        self.outbox = defaultdict(lambda: [])

    def schedule_message(self, sender, recepient, contents):
        """
        Moves an item to an agent's outbox. It will be sent the next time
        carry_mail is called.
        """
        assert(recepient in self.contacts[sender])
        self.outbox[sender].append(self.Message(sender, recepient, contents))

    def carry_mail(self):
        """
        Routes each message from its sender's outbox to its recepient's inbox.
        All outboxes should be empty after this funtion runs.
        """
        for sender, messages in self.outbox.items():
            for message in messages:
                self.send_message(sender, message.recepient, message.contents)
            self.clear_outbox(sender)
        
    def send_message(self, sender, recepient, contents):
        """
        Moves an item into an agent's inbox, if the contact is permitted
        """
        assert(recepient in self.contacts[sender])
        self.inbox[recepient].append(self.Message(sender, recepient, contents))
        
    def get_inbox_messages(self):
        """
        Returns a list of messages in the format (sender, message)
        """
        msg_list = []
        for sender, messages in self.inbox.items():
            for message in messages:
                msg_list.append((sender, message))
        
        return msg_list
        
    def clear_inbox(self, id):
        self.inbox[id] = []
        
    def clear_outbox(self, id):
        self.outbox[id] = []
        
        
