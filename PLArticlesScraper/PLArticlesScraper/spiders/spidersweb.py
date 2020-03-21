# -*- coding: utf-8 -*-
import scrapy


class SpiderswebSpider(scrapy.Spider):
    name = 'spidersweb'
    allowed_domains = ['spidersweb.pl', 'bezprawnik.pl']
    start_urls = ['http://spidersweb.pl/']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//article[contains(@class, 'article')]//a/@href"
    NEXT_PAGE_URL = "//a[contains(@class, 'nextpostslink')]/@href"

    TITLE_NODE = "//h1"
    FIRST_TYPE_LEAD = "//div[contains(@class, 'article-main')]//strong/text()"
    FIRST_TYPE_TEXT = "//div[contains(@class, 'article-main')]/p[not(img)] | //div[contains(@class, 'article-main')]/h3"

    SECOND_TYPE_LEAD = "//section[contains(@class, 'content')]//strong/text()"
    SECOND_TYPE_TEXT = "//section[contains(@class, 'content')]/p[not(img)] | //section[contains(@class, 'content')]/h3"

    THIRD_TYPE_LEAD = "//div[contains(@class, 'text-content')]//strong/text()"
    THIRD_TYPE_TEXT = "//div[contains(@class, 'text-content')]/p[not(img)] | //div[contains(@class, 'text-content')]/h3"

    def parse(self, response):
        # iterate through links to articles
        for quote in response.xpath(self.ARTICLES_URLS):
            yield response.follow(quote, callback=self.parse_article)

        #iterate through next page links
        for next_page in response.xpath(self.NEXT_PAGE_URL):
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract_first(),
            'lead': [
            	*response.xpath(self.FIRST_TYPE_LEAD).extract(),
            	*response.xpath(self.SECOND_TYPE_LEAD).extract(),
            	*response.xpath(self.THIRD_TYPE_LEAD).extract(),
            ],
            'text': [
            	*response.xpath(self.FIRST_TYPE_TEXT).extract(),
            	*response.xpath(self.SECOND_TYPE_TEXT).extract(),
            	*response.xpath(self.THIRD_TYPE_TEXT).extract(),
            ]
        }