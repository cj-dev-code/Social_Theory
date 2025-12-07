# src/feminism_rag/discord_bot.py
from __future__ import annotations

import discord
from discord.ext import commands

from .generate import graph
from . import config

MAG = "🔍"
RECYCLE = "♻️"


class DictQueue(dict):
    """Very small queue-like cache for (message_id -> response) pairs."""

    def __init__(self, maxlen: int = 5):
        self.list = []
        self.maxlen = maxlen

    def __getitem__(self, message_id: int):
        for mid, resp in self.list:
            if mid == message_id:
                return resp
        return {}

    def __setitem__(self, message_id: int, response: dict):
        # push to front, drop the oldest (end) if > maxlen
        self.list.insert(0, (message_id, response))
        if len(self.list) > self.maxlen:
            self.list.pop()


cache = DictQueue(maxlen=5)

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.reactions = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.command(name="askchima")
async def askchima(ctx, *, text: str):
    """Usage: !askchima <your question>"""

    try:
        result = graph.invoke({"question": text})
    except Exception as e:
        msg = await ctx.send(f"backend error: {e}")
        try:
            await msg.add_reaction(RECYCLE)
        except Exception:
            pass
        return

    answer = result.get("answer", "(no answer)")
    msg = await ctx.send(f"**Answer:** {answer}")
    try:
        await msg.add_reaction(MAG)
    except Exception:
        pass

    cache[msg.id] = result  # store full response (incl. context)


@bot.command(name="chimahelp")
async def chimahelp(ctx):
    """Usage: !chimahelp"""
    message = """
You can do this:
- `!chimahelp` for this message
- `!askchima <your question>` e.g. `!askchima why does simone think women are the second sex?` for a question about the books in the booklist!
- `!booklist` for the books you can ask about
"""
    msg = await ctx.send(message)
    try:
        await msg.add_reaction(RECYCLE)
    except Exception:
        pass


@bot.command(name="booklist")
async def booklist(ctx):
    """Usage: !booklist"""
    message = """
This bot talks about:
- abortion: our bodies, their lies, and the truths we use to win, jess valenti [[goodreads]](<https://www.goodreads.com/book/show/210246742-abortion>)
- algorithms of oppression: how search engines reinforce racism, safiya umoja noble [[goodreads]](<https://www.goodreads.com/book/show/34762552-algorithms-of-oppression>)
- black feminist thought, patricia hill [[goodreads]](<https://www.goodreads.com/book/show/353598.Black_Feminist_Thought>)
- data feminism, catherine d'ignazio and lauren klein [[goodreads]](<https://www.goodreads.com/book/show/51777543-data-feminism>)
- invisible women: exposing data bias in a world designed for men, caroline perez [[goodreads]](<https://www.goodreads.com/book/show/41104077-invisible-women>)
- the second sex, simone de beauvoir [[goodreads]](<https://www.goodreads.com/book/show/457264.The_Second_Sex>)
- the will to change: men, masculinity, and love, by bell hooks [[goodreads]](<https://www.goodreads.com/book/show/17601.The_Will_to_Change>)
- we should all be feminists, chimamanda ngozi adichie [[goodreads]](<https://www.goodreads.com/book/show/22738563-we-should-all-be-feminists>)
- white feminism - koa beck [[goodreads]](<https://www.goodreads.com/book/show/54238294-white-feminism>)
- why does he do that? inside the minds of angry men - lundy bancroft [[goodreads]](<https://www.goodreads.com/book/show/224552.Why_Does_He_Do_That_>)
- women, race and class - angela davis [[goodreads]](<https://www.goodreads.com/book/show/635635.Women_Race_Class>)
"""
    msg = await ctx.send(message)
    try:
        await msg.add_reaction(RECYCLE)
    except Exception:
        pass


@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return
    if str(reaction.emoji) not in (MAG, RECYCLE):
        return

    msg = reaction.message
    if msg.author.id != bot.user.id:
        return

    if str(reaction.emoji) == MAG:
        data = cache[msg.id]
        if not data:
            await msg.channel.send(
                "sorry, that message is too old. please ask the question again and react with a mag for context. thanks!",
                reference=msg,
            )
            return

        docs = data.get("context", [])
        lines = [f"{MAG} **Context for message ID `{msg.id}`**"]
        for i, doc in enumerate(docs, 1):
            try:
                source = doc.metadata["title"] + " - " + doc.metadata["author"]
            except Exception:
                if "Black Feminist Thought" in repr(getattr(doc, "metadata", {})):
                    source = "Black Feminist Thought - Patricia Hill"
                else:
                    source = "error, oops! ping the maintainer!"
            try:
                page = getattr(doc, "page_content", "")
            except Exception:
                page = "error! oops!"
            lines.append(
                f"\n**[{i}] Source:** `{source}`\n**Page content:**\n{page}"[:380]
                + "..."
            )

        output = await msg.channel.send(
            "\n".join(lines)[:2000],
            reference=msg,
        )
        try:
            await output.add_reaction(RECYCLE)
        except Exception:
            pass
    elif str(reaction.emoji) == RECYCLE:
        await msg.delete()


def main() -> None:
    """Entry point for running the Discord bot."""
    if not config.DISCORD_BOT_TOKEN:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is not set. Add it to your .env before running."
        )
    bot.run(config.DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
