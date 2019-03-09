# -*- coding: utf-8 -*-

"""Module that contains utilities useful for dealing with emojis in text."""

KNOWN_EMOJIS = {
  ':)': 'smile',
  ':-)': 'smile',
  ':(': 'sad',
  ':-(': 'sad',
  '&lt;3': 'heart',
  '>:(': 'angry',
  '>:-(': 'angry',
  ':\'(': 'cry',
}


def replace_emojis(orig_text):
  """
  Replace known emoji character combinations with words. Doing the replacement
  so that in later pre-processing stages, emojis are not suppressed as
  punctuation.

  note:: The transformation is **not** done in place. Input remains intact.

  :param orig_text: Original line of text to transform.
  :type orig_text: str
  :return: New line of text with transformation done.
  :rtype: str
  """
  result = '{:s}'.format(orig_text)
  for emoji, subst in KNOWN_EMOJIS.items():
    result = result.replace(emoji, subst)
  return result

# vim: set ts=2 sw=2 expandtab:
